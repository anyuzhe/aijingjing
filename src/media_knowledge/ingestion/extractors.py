from __future__ import annotations

import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

from ..models import ContentSegment
from ..product import DesktopSettings, ProductPaths
from .types import CancellationToken, ExtractionResult
from .vision import MultimodalInterpreter


class MissingExtractorDependency(RuntimeError):
    pass


@dataclass(slots=True)
class ExtractionContext:
    paths: ProductPaths
    settings: DesktopSettings
    cancellation: CancellationToken
    vision: MultimodalInterpreter | None = None
    progress: Callable[[str], None] | None = None

    def message(self, value: str) -> None:
        self.cancellation.check()
        if self.progress:
            self.progress(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def safe_stem(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    compact = re.sub(r"[/:*?\"<>|\\]", "-", compact).strip(" .-")
    return compact[:96] or "未命名资料"


class TextExtractor:
    suffixes = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".py", ".js", ".ts"}

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        context.message("正在读取文本")
        text = path.read_text(encoding="utf-8")
        return ExtractionResult(
            title=path.stem,
            media_type="markdown" if path.suffix.casefold() in {".md", ".markdown"} else "text",
            segments=[ContentSegment("text-1", 1, "text", text=text)],
            source_path=path,
            checksum=sha256_file(path),
        )


class PDFExtractor:
    suffixes = {".pdf"}

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        try:
            try:
                import pymupdf as fitz  # type: ignore
            except ImportError:
                import fitz  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("PDF 解析组件 PyMuPDF 未安装") from exc
        context.message("正在逐页解析 PDF")
        document = fitz.open(path)
        page_count = len(document)
        segments: list[ContentSegment] = []
        warnings: list[str] = []
        assets: list[Path] = []
        title = str(document.metadata.get("title") or "").strip() or path.stem
        asset_dir = context.paths.assets / "pdf" / safe_stem(path.stem)
        for index, page in enumerate(document):
            context.cancellation.check()
            page_number = index + 1
            text = page.get_text("text").strip()
            description = ""
            should_render = page_number == 1 or len(text) < 100
            image_path: Path | None = None
            if should_render:
                try:
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    image_path = asset_dir / f"page-{page_number:04d}.png"
                    page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(image_path)
                    assets.append(image_path)
                except Exception as exc:
                    warnings.append(f"第 {page_number} 页预览保留失败：{type(exc).__name__}")
            if image_path and context.vision and context.vision.available:
                try:
                    description = context.vision.describe(image_path, context=text[:3000])
                except Exception as exc:
                    warnings.append(f"第 {page_number} 页视觉分析失败：{type(exc).__name__}")
            if not text and not description:
                warnings.append(f"第 {page_number} 页没有提取到文字；可能需要 OCR")
                continue
            segments.append(
                ContentSegment(
                    f"page-{page_number}",
                    page_number,
                    "page",
                    text=text,
                    description=description,
                    location={"page": page_number},
                    asset=str(image_path) if image_path else None,
                )
            )
        document.close()
        return ExtractionResult(
            title=title,
            media_type="pdf",
            segments=segments,
            source_path=path,
            checksum=sha256_file(path),
            warnings=warnings,
            retained_assets=assets,
            metadata={"page_count": page_count},
        )


class DOCXExtractor:
    suffixes = {".docx"}

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("Word 解析组件 python-docx 未安装") from exc
        context.message("正在解析 Word 结构")
        document = Document(path)
        segments: list[ContentSegment] = []
        heading_path: list[str] = []
        sequence = 0
        for paragraph in document.paragraphs:
            context.cancellation.check()
            value = paragraph.text.strip()
            if not value:
                continue
            style = str(paragraph.style.name or "")
            heading = re.match(r"Heading\s+(\d+)", style, re.IGNORECASE)
            if heading:
                level = int(heading.group(1))
                heading_path = heading_path[: level - 1] + [value]
                continue
            sequence += 1
            segments.append(ContentSegment(f"paragraph-{sequence}", sequence, "text", text=value, heading_path=list(heading_path)))
        for table_index, table in enumerate(document.tables, 1):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
            if rows:
                sequence += 1
                segments.append(
                    ContentSegment(
                        f"table-{table_index}", sequence, "table", text="\n".join(rows),
                        description=json.dumps({"rows": len(rows)}, ensure_ascii=False),
                        heading_path=list(heading_path),
                    )
                )
        core_title = str(document.core_properties.title or "").strip()
        return ExtractionResult(
            title=core_title or path.stem,
            media_type="document",
            segments=segments,
            source_path=path,
            checksum=sha256_file(path),
            metadata={"paragraphs": len(document.paragraphs), "tables": len(document.tables)},
        )


class PPTXExtractor:
    suffixes = {".pptx"}

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        try:
            from pptx import Presentation  # type: ignore
            from pptx.enum.shapes import MSO_SHAPE_TYPE  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("PPTX 解析组件 python-pptx 未安装") from exc
        context.message("正在逐页解析演示文稿")
        presentation = Presentation(path)
        segments: list[ContentSegment] = []
        assets: list[Path] = []
        warnings: list[str] = []
        asset_dir = context.paths.assets / "slides" / safe_stem(path.stem)
        slide_total = len(presentation.slides)
        for slide_index, slide in enumerate(presentation.slides, 1):
            context.cancellation.check()
            texts: list[str] = []
            descriptions: list[str] = []
            has_visual_structure = False
            for shape_index, shape in enumerate(slide.shapes, 1):
                if getattr(shape, "has_text_frame", False):
                    value = str(shape.text or "").strip()
                    if value:
                        texts.append(value)
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    has_visual_structure = True
                    try:
                        asset_dir.mkdir(parents=True, exist_ok=True)
                        image = shape.image
                        image_path = asset_dir / f"slide-{slide_index:04d}-image-{shape_index:02d}.{image.ext}"
                        image_path.write_bytes(image.blob)
                        assets.append(image_path)
                    except Exception as exc:
                        warnings.append(f"第 {slide_index} 页图片提取失败：{type(exc).__name__}")
                if getattr(shape, "has_table", False):
                    has_visual_structure = True
                if getattr(shape, "has_chart", False):
                    has_visual_structure = True
                    chart_text = self._chart_text(shape)
                    if chart_text:
                        texts.append(chart_text)
            notes = ""
            try:
                notes_frame = slide.notes_slide.notes_text_frame
                notes = str(notes_frame.text or "").strip() if notes_frame else ""
            except (AttributeError, KeyError, ValueError):
                pass
            content = "\n\n".join(texts)
            if notes:
                content += ("\n\n演讲备注：\n" if content else "演讲备注：\n") + notes
            preview_path: Path | None = None
            important_words = ("架构", "流程", "系统", "结果", "曲线", "参数", "对比", "architecture", "flow", "result")
            important = slide_total <= 60 or has_visual_structure or any(word in content.casefold() for word in important_words)
            if important:
                try:
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    preview_path = asset_dir / f"slide-{slide_index:04d}.png"
                    self._render_preview(presentation, slide, preview_path)
                    assets.append(preview_path)
                except Exception as exc:
                    warnings.append(f"第 {slide_index} 页预览保留失败：{type(exc).__name__}")
            if preview_path and context.vision and context.vision.available and (has_visual_structure or len(content) < 180):
                try:
                    descriptions.append(context.vision.describe(preview_path, context=content[:4000]))
                except Exception as exc:
                    warnings.append(f"第 {slide_index} 页视觉分析失败：{type(exc).__name__}")
            if not content and not descriptions:
                warnings.append(f"第 {slide_index} 页没有可索引内容")
                continue
            segments.append(
                ContentSegment(
                    f"slide-{slide_index}", slide_index, "slide", text=content,
                    description="\n\n".join(descriptions), location={"slide": slide_index},
                    heading_path=[texts[0].splitlines()[0][:120]] if texts else [],
                    asset=str(preview_path) if preview_path else None,
                )
            )
        return ExtractionResult(
            title=path.stem,
            media_type="presentation",
            segments=segments,
            source_path=path,
            checksum=sha256_file(path),
            warnings=warnings,
            retained_assets=assets,
            metadata={"slide_count": len(presentation.slides)},
        )

    @staticmethod
    def _chart_text(shape) -> str:
        try:
            chart = shape.chart
            values = []
            if chart.has_title:
                values.append(f"图表标题：{chart.chart_title.text_frame.text}")
            for series in chart.series:
                raw = list(series.values)
                values.append(f"数据系列 {series.name}：{raw[:30]}")
            return "\n".join(values)
        except (AttributeError, TypeError, ValueError):
            return ""

    @staticmethod
    def _render_preview(presentation, slide, destination: Path) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("PPT 页面预览需要 Pillow") from exc
        width = 1600
        height = max(900, round(width * presentation.slide_height / presentation.slide_width))
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        scale_x = width / presentation.slide_width
        scale_y = height / presentation.slide_height
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        font_path = next((value for value in font_paths if Path(value).is_file()), None)

        def font(size: int):
            return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()

        for shape in slide.shapes:
            x = round(shape.left * scale_x)
            y = round(shape.top * scale_y)
            w = max(1, round(shape.width * scale_x))
            h = max(1, round(shape.height * scale_y))
            if getattr(shape, "shape_type", None) == 13 and hasattr(shape, "image"):
                try:
                    picture = Image.open(io.BytesIO(shape.image.blob)).convert("RGB")
                    picture.thumbnail((w, h))
                    canvas.paste(picture, (x + (w - picture.width) // 2, y + (h - picture.height) // 2))
                    continue
                except (OSError, ValueError):
                    pass
            draw.rounded_rectangle((x, y, min(width - 1, x + w), min(height - 1, y + h)), radius=6, outline="#d6ddd8", width=2)
            text = str(getattr(shape, "text", "") or "").strip()
            if getattr(shape, "has_table", False):
                text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in shape.table.rows)
            if text:
                size = max(16, min(38, round(h / max(3, text.count("\n") + 2))))
                max_chars = max(8, round(w / max(8, size * 0.72)))
                lines = []
                for paragraph in text.splitlines():
                    lines.extend(paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars))
                draw.multiline_text((x + 8, y + 6), "\n".join(lines[:18]), fill="#26322d", font=font(size), spacing=5)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, optimize=True)


class ImageExtractor:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("图片组件 Pillow 未安装") from exc
        context.message("正在识别图片内容")
        warnings: list[str] = []
        ocr = ""
        with Image.open(path) as image:
            metadata = {"width": image.width, "height": image.height, "format": image.format}
            try:
                try:
                    from rapidocr import RapidOCR  # type: ignore
                except ImportError:
                    from rapidocr_onnxruntime import RapidOCR  # type: ignore

                raw_result = RapidOCR()(str(path))
                if hasattr(raw_result, "txts"):
                    result = [[None, text] for text in (raw_result.txts or [])]
                else:
                    result = raw_result[0]
                if result:
                    ocr = "\n".join(str(item[1]).strip() for item in result if len(item) > 1 and str(item[1]).strip())
            except (ImportError, RuntimeError, OSError, ValueError):
                pass
            try:
                import pytesseract  # type: ignore

                if not ocr:
                    ocr = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
            except (ImportError, RuntimeError, OSError):
                if not ocr:
                    warnings.append("OCR 不可用；已尝试多模态视觉理解")
        description = ""
        if context.vision and context.vision.available:
            try:
                description = context.vision.describe(path, context=ocr)
            except Exception as exc:
                warnings.append(f"视觉分析失败：{type(exc).__name__}")
        if not ocr and not description:
            raise MissingExtractorDependency("图片没有可索引文字，且未配置可用的视觉模型或 OCR")
        return ExtractionResult(
            title=path.stem,
            media_type="image",
            segments=[ContentSegment("image-1", 1, "image", text=ocr, description=description, asset=str(path))],
            source_path=path,
            checksum=sha256_file(path),
            warnings=warnings,
            metadata=metadata,
        )


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class AudioVideoExtractor:
    audio_suffixes = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus"}
    video_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    suffixes = audio_suffixes | video_suffixes

    def extract(self, path: Path, context: ExtractionContext) -> ExtractionResult:
        ffmpeg = _ffmpeg_executable()
        if not ffmpeg:
            raise MissingExtractorDependency("音视频组件 FFmpeg 未安装或未随应用打包")
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency("语音识别组件 faster-whisper 未安装") from exc
        is_video = path.suffix.casefold() in self.video_suffixes
        context.message("正在提取音轨")
        with tempfile.TemporaryDirectory(prefix="ai-jingjing-media-") as temporary:
            audio_path = Path(temporary) / "audio.wav"
            process = subprocess.run(
                [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
                capture_output=True,
                timeout=30 * 60,
            )
            if process.returncode != 0 or not audio_path.is_file():
                raise RuntimeError("FFmpeg 无法提取音轨")
            context.message("正在进行语音识别")
            model = WhisperModel(context.settings.whisper_model, device="cpu", compute_type="int8")
            transcribed, info = model.transcribe(str(audio_path), vad_filter=True)
            segments: list[ContentSegment] = []
            transcript_lines: list[str] = []
            for index, item in enumerate(transcribed, 1):
                context.cancellation.check()
                text = str(item.text or "").strip()
                if not text:
                    continue
                transcript_lines.append(f"[{item.start:.2f}-{item.end:.2f}] {text}")
                segments.append(
                    ContentSegment(
                        f"speech-{index}", item.start, "speech", text=text,
                        location={"timestamp_start": float(item.start), "timestamp_end": float(item.end)},
                        metadata={"language": getattr(info, "language", None)},
                    )
                )
            transcript_path = context.paths.transcripts / f"{safe_stem(path.stem)}-{sha256_file(path)[:10]}.txt"
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
            assets: list[Path] = []
            warnings: list[str] = []
            if is_video and context.vision and context.vision.available:
                context.message("正在抽取视频关键帧")
                frame_dir = context.paths.assets / "frames" / safe_stem(path.stem)
                frame_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        ffmpeg, "-y", "-i", str(path), "-vf", "fps=1/60,scale=1280:-2",
                        "-frames:v", "8", str(frame_dir / "frame-%03d.jpg"),
                    ],
                    capture_output=True,
                    timeout=30 * 60,
                )
                for frame_index, frame in enumerate(sorted(frame_dir.glob("frame-*.jpg")), 1):
                    try:
                        assets.append(frame)
                        description = context.vision.describe(frame, context="视频关键帧")
                        segments.append(
                            ContentSegment(
                                f"frame-{frame_index}", frame_index * 60, "image",
                                description=description,
                                location={"timestamp_start": float((frame_index - 1) * 60)},
                                asset=str(frame),
                            )
                        )
                    except Exception as exc:
                        warnings.append(f"关键帧 {frame_index} 分析失败：{type(exc).__name__}")
        if not segments:
            raise RuntimeError("音视频没有提取到可索引内容")
        return ExtractionResult(
            title=path.stem,
            media_type="video" if is_video else "audio",
            segments=segments,
            source_path=path,
            checksum=sha256_file(path),
            warnings=warnings,
            retained_assets=assets,
            transcript_path=transcript_path,
        )


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._ignored = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") in {"og:title", "twitter:title"}:
            self.title = str(attributes.get("content") or self.title)
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored:
            return
        if self._in_title:
            self.title += data
        else:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


class _WeixinArticleHTML(HTMLParser):
    """Extract the real article body from a Weixin public-account page."""

    _void_tags = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
    _block_tags = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "blockquote", "br", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._content_depth = 0
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("property") in {"og:title", "twitter:title"}:
            self.title = str(attributes.get("content") or self.title)
        entering_content = attributes.get("id") == "js_content"
        if entering_content:
            self._content_depth = 1
        elif self._content_depth and tag not in self._void_tags:
            self._content_depth += 1
        if self._content_depth and tag in {"script", "style", "noscript", "svg"}:
            self._ignored += 1
        if self._content_depth and tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._content_depth:
            return
        if tag in {"script", "style", "noscript", "svg"} and self._ignored:
            self._ignored -= 1
        if tag in self._block_tags:
            self.parts.append("\n")
        if tag not in self._void_tags:
            self._content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._content_depth and not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r"\n\s*\n+", "\n\n", value)
        return value.strip()


def _download_web_document(url: str, context: ExtractionContext) -> tuple[bytes, str, str]:
    context.message("正在下载网页快照")
    request = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(15 * 1024 * 1024 + 1)
        if len(raw) > 15 * 1024 * 1024:
            raise ValueError("网页超过 15MB 安全限制")
        charset = response.headers.get_content_charset() or "utf-8"
        document = raw.decode(charset, errors="replace")
        final_url = response.geturl() if hasattr(response, "geturl") else url
    return raw, document, final_url


def _raise_for_weixin_challenge(final_url: str, document: str) -> None:
    path = urllib.parse.urlsplit(final_url).path.casefold()
    challenge_markers = (
        "当前环境异常，完成验证后即可继续访问",
        "wappoc_appmsgcaptcha",
        "访问过于频繁，请用微信扫描二维码进行访问",
    )
    if path.startswith("/mp/wappoc_appmsgcaptcha") or any(marker in document for marker in challenge_markers):
        raise RuntimeError("微信返回了访问验证页，没有取得文章正文；请稍后重试或在浏览器中打开后导出为 PDF 再导入")


class WebExtractor:
    def extract(self, url: str, context: ExtractionContext) -> ExtractionResult:
        raw, document, _final_url = _download_web_document(url, context)
        parser = _ReadableHTML()
        parser.feed(document)
        text = parser.text()
        if not text:
            raise RuntimeError("网页没有提取到正文；可能需要登录或动态浏览器渲染")
        return ExtractionResult(
            title=parser.title.strip() or url,
            media_type="web",
            segments=[ContentSegment("web-1", 1, "text", text=text)],
            original_uri=url,
            checksum=hashlib.sha256(raw).hexdigest(),
            metadata={"snapshot_html": document},
        )


class WeixinArticleExtractor:
    """Read public Weixin articles without treating anti-bot pages as knowledge."""

    @classmethod
    def supports(cls, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").casefold()
        return parsed.scheme.casefold() in {"http", "https"} and host == "mp.weixin.qq.com" and parsed.path.startswith("/s/")

    def extract(self, url: str, context: ExtractionContext) -> ExtractionResult:
        raw, document, final_url = _download_web_document(url, context)
        _raise_for_weixin_challenge(final_url, document)
        parser = _WeixinArticleHTML()
        parser.feed(document)
        text = parser.text()
        if len(text) < 80:
            raise RuntimeError("微信公众号页面没有提取到足够的文章正文；页面可能已删除、需要验证或限制外部访问")
        return ExtractionResult(
            title=parser.title.strip() or url,
            media_type="web",
            segments=[ContentSegment("web-1", 1, "text", text=text)],
            original_uri=url,
            checksum=hashlib.sha256(raw).hexdigest(),
            metadata={
                "snapshot_html": document,
                "platform": "weixin_public_account",
                "content_scope": "full_source",
            },
        )


class DirectMediaURLExtractor:
    """Download a public direct audio/video URL, then run the local media pipeline."""

    media_suffixes = AudioVideoExtractor.suffixes
    max_download_bytes = 2 * 1024 * 1024 * 1024
    content_type_suffixes = {
        "audio/aac": ".aac",
        "audio/flac": ".flac",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-m4v": ".m4v",
        "video/x-matroska": ".mkv",
        "video/x-msvideo": ".avi",
    }

    @classmethod
    def supports(cls, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        return parsed.scheme.casefold() in {"http", "https"} and Path(parsed.path).suffix.casefold() in cls.media_suffixes

    def extract(self, url: str, context: ExtractionContext) -> ExtractionResult:
        return self.extract_remote(url, context, original_uri=url)

    def extract_remote(
        self,
        media_url: str,
        context: ExtractionContext,
        *,
        original_uri: str,
        title: str | None = None,
        metadata: dict[str, object] | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> ExtractionResult:
        parsed = urllib.parse.urlsplit(media_url)
        if parsed.scheme.casefold() not in {"http", "https"}:
            raise ValueError("远程音视频地址必须使用 HTTP 或 HTTPS")
        headers = {"User-Agent": "Mozilla/5.0 AI-Jingjing/1.0", **(request_headers or {})}
        request = urllib.request.Request(media_url, headers=headers)
        context.message("正在下载远程音视频")
        with urllib.request.urlopen(request, timeout=90) as response:
            content_type = str(response.headers.get_content_type() or "").casefold()
            declared_length = response.headers.get("Content-Length")
            if declared_length:
                try:
                    if int(declared_length) > self.max_download_bytes:
                        raise ValueError("远程音视频超过 2GB 安全限制，请先下载后再导入")
                except ValueError as exc:
                    if "超过" in str(exc):
                        raise
            suffix = Path(parsed.path).suffix.casefold()
            if suffix not in self.media_suffixes:
                suffix = self.content_type_suffixes.get(content_type, "")
            if not (content_type.startswith(("audio/", "video/")) or suffix in self.media_suffixes):
                raise ValueError("该地址不是可直接下载的音视频链接")
            if suffix not in self.media_suffixes:
                suffix = ".mp4" if content_type.startswith("video/") else ".m4a"
            source_stem = safe_stem(title or Path(parsed.path).stem or "远程音视频")
            identity = hashlib.sha256(original_uri.encode("utf-8")).hexdigest()[:12]
            destination = context.paths.cache / "remote-media" / f"{source_stem}-{identity}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            downloaded = 0
            try:
                with temporary.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        context.cancellation.check()
                        downloaded += len(block)
                        if downloaded > self.max_download_bytes:
                            raise ValueError("远程音视频超过 2GB 安全限制，请先下载后再导入")
                        handle.write(block)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        extracted = AudioVideoExtractor().extract(destination, context)
        extracted.title = title or extracted.title
        extracted.original_uri = original_uri
        extracted.metadata.update(
            {
                "remote_media": True,
                "remote_content_type": content_type,
                "remote_download_bytes": downloaded,
                **(metadata or {}),
            }
        )
        return extracted


class WeixinChannelsExtractor:
    """Resolve public Weixin Channels share metadata and playable streams when exposed."""

    api_url = "https://channels.weixin.qq.com/finder-preview/api/feed/get_feed_info"
    allowed_hosts = {"weixin.qq.com", "www.weixin.qq.com", "channels.weixin.qq.com"}

    @classmethod
    def supports(cls, url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme.casefold() not in {"http", "https"} or host not in cls.allowed_hosts:
            return False
        return parsed.path.startswith("/sph/") or parsed.path.startswith("/finder-preview/pages/sph")

    @staticmethod
    def _short_uri(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        if parsed.path.startswith("/sph/"):
            candidate = parsed.path.split("/sph/", 1)[1].split("/", 1)[0]
        else:
            candidate = urllib.parse.parse_qs(parsed.query).get("id", [""])[0]
        candidate = urllib.parse.unquote(candidate).strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{5,128}", candidate):
            raise ValueError("无法识别微信视频号短链编号")
        return candidate

    @staticmethod
    def _playable_url(feed: dict[str, object]) -> str | None:
        for key in ("h264VideoInfo", "h265VideoInfo"):
            value = feed.get(key)
            if isinstance(value, dict) and isinstance(value.get("videoUrl"), str) and value["videoUrl"]:
                return str(value["videoUrl"])
        value = feed.get("videoUrl")
        return str(value) if isinstance(value, str) and value else None

    @staticmethod
    def _title(description: str, author: str) -> str:
        first = re.split(r"[。！？\n]", description, maxsplit=1)[0].strip()
        base = first[:72] or "微信视频号内容"
        return f"{base}｜{author}" if author else base

    def extract(self, url: str, context: ExtractionContext) -> ExtractionResult:
        short_uri = self._short_uri(url)
        preview_url = f"https://channels.weixin.qq.com/finder-preview/pages/sph?id={urllib.parse.quote(short_uri)}"
        payload = json.dumps(
            {"baseReq": {"generalToken": ""}, "shortUri": short_uri},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://channels.weixin.qq.com",
                "Referer": preview_url,
                "User-Agent": "Mozilla/5.0 AI-Jingjing/1.0",
            },
        )
        context.message("正在解析微信视频号分享链接")
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"微信视频号链接解析失败（HTTP {exc.code}）") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("暂时无法连接微信视频号，请稍后重试") from exc
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("微信视频号返回数据超过安全限制")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("微信视频号返回了无法识别的数据") from exc
        data = result.get("data") if isinstance(result, dict) else None
        data = data if isinstance(data, dict) else {}
        feed = data.get("feedInfo")
        feed = feed if isinstance(feed, dict) else {}
        author_info = data.get("authorInfo")
        author_info = author_info if isinstance(author_info, dict) else {}
        description = str(feed.get("description") or "").strip()
        author = str(author_info.get("nickname") or "").strip()
        error_info = data.get("errMsg")
        if not feed and isinstance(error_info, dict):
            reason = str(error_info.get("title") or "此内容暂时无法读取")
            raise RuntimeError(
                f"{reason}。内容可能需要微信登录、仅好友可见、已删除或链接已过期；"
                "请在微信中保存视频文件后导入。"
            )
        title = self._title(description, author)
        platform_metadata: dict[str, object] = {
            "platform": "weixin_channels",
            "platform_author": author,
            "platform_description": description,
            "platform_cover_url": str(feed.get("coverUrl") or ""),
            "platform_created_at": feed.get("createtime"),
            "weixin_short_uri": short_uri,
        }
        playable_url = self._playable_url(feed)
        if playable_url:
            context.message("已取得视频流，准备下载并转写")
            return DirectMediaURLExtractor().extract_remote(
                playable_url,
                context,
                original_uri=url,
                title=title,
                metadata={**platform_metadata, "content_scope": "full_media"},
                request_headers={"Referer": preview_url},
            )
        raise RuntimeError(
            "微信视频号公开分享页只提供说明和封面，没有开放真实音视频流；"
            "为避免把简介误当成视频内容，本次不入库。请在微信中保存原视频文件后拖入导入。"
        )


def url_extractor_for(url: str):
    if WeixinChannelsExtractor.supports(url):
        return WeixinChannelsExtractor()
    if WeixinArticleExtractor.supports(url):
        return WeixinArticleExtractor()
    if DirectMediaURLExtractor.supports(url):
        return DirectMediaURLExtractor()
    return WebExtractor()


FILE_EXTRACTORS = [TextExtractor(), PDFExtractor(), DOCXExtractor(), PPTXExtractor(), ImageExtractor(), AudioVideoExtractor()]


def extractor_for(path: Path):
    suffix = path.suffix.casefold()
    return next((extractor for extractor in FILE_EXTRACTORS if suffix in extractor.suffixes), None)
