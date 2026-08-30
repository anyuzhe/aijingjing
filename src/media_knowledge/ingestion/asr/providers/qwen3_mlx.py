from __future__ import annotations

import importlib.util
import platform
import re
from importlib import import_module
from pathlib import Path

from ...types import CancelledError
from ..base import AsrProviderError, CancellationCallback, ProgressCallback
from ..types import AsrResult, AsrSegment, TranscriptionRequest
from ._shared import normalize_language, parse_segments


class Qwen3MlxProvider:
    """Qwen3-ASR via mlx-audio, with strictly local model loading.

    ``mlx_audio.stt.load`` accepts either a repository ID or a local path.  This
    provider deliberately supplies a :class:`Path`, never a repository ID, so a
    transcription job cannot initiate a model download.
    """

    provider_id = "qwen3-mlx"
    _MAX_TOKENS = 8192
    _STREAM_CHUNK_SECONDS = 60.0

    def __init__(self, *, available_override: bool | None = None) -> None:
        self._available_override = available_override

    def available(self) -> bool:
        if self._available_override is not None:
            return bool(self._available_override)
        apple = platform.system().casefold() == "darwin" and platform.machine().casefold() in {
            "arm64", "aarch64",
        }
        try:
            installed = importlib.util.find_spec("mlx_audio") is not None
        except (ImportError, AttributeError, ValueError):
            installed = False
        return apple and installed

    @staticmethod
    def _language(value: str | None) -> str | None:
        normalized = str(value or "").strip().casefold()
        if not normalized or normalized in {"auto", "automatic", "自动"}:
            return None
        aliases = {
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "中文": "Chinese",
            "mandarin": "Chinese",
            "yue": "Cantonese",
            "粤语": "Cantonese",
            "en": "English",
            "英语": "English",
            "ja": "Japanese",
            "日语": "Japanese",
            "ko": "Korean",
            "韩语": "Korean",
        }
        return aliases.get(normalized, value)

    @staticmethod
    def _local_model_path(request: TranscriptionRequest) -> Path:
        candidate = request.model_path
        if candidate is None:
            raw = Path(str(request.model or "")).expanduser()
            candidate = raw if raw.exists() else None
        if candidate is None:
            raise AsrProviderError(
                "Qwen3-ASR 必须先选择已下载的本地模型目录",
                reason_code="model_not_local",
            )
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise AsrProviderError(
                "Qwen3-ASR 本地模型目录不存在或不可访问",
                reason_code="model_not_local",
            ) from None
        if not resolved.is_dir() or not (resolved / "config.json").is_file():
            raise AsrProviderError(
                "Qwen3-ASR 本地模型目录缺少 config.json",
                reason_code="model_invalid",
            )
        return resolved

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> AsrResult:
        if check_cancelled:
            check_cancelled()
        local_model = self._local_model_path(request)
        try:
            stt = import_module("mlx_audio.stt")
        except ImportError as exc:
            raise AsrProviderError(
                "mlx-audio 未安装，无法运行 Qwen3-ASR",
                reason_code="dependency_missing",
            ) from exc
        if check_cancelled:
            check_cancelled()
        if progress:
            progress(f"正在加载本地 Qwen3-ASR 模型：{local_model.name}")
        try:
            # Passing Path (rather than str) is intentional: mlx-audio only
            # resolves/downloads a Hub model when the argument is a string.
            model = stt.load(local_model)
        except CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AsrProviderError(
                f"Qwen3-ASR 本地模型加载失败（{type(exc).__name__}）",
                reason_code="model_load_failed",
            ) from exc
        if check_cancelled:
            check_cancelled()
        if progress:
            progress("本地 Qwen3-ASR 模型已加载，正在进行 MLX 转写")
        kwargs: dict[str, object] = {
            "max_tokens": self._MAX_TOKENS,
            "temperature": 0.0,
            # mlx-audio yields between decoder tokens in streaming mode, so a
            # user cancellation can stop the generation loop promptly.
            "stream": True,
            "chunk_duration": self._STREAM_CHUNK_SECONDS,
        }
        forced_language = self._language(request.language)
        if forced_language:
            kwargs["language"] = forced_language
        if request.context_terms:
            # Current mlx-audio folds hotwords into Qwen3-ASR's system prompt.
            kwargs["hotwords"] = list(request.context_terms)
        try:
            output = model.generate(str(request.audio_path), **kwargs)
        except CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AsrProviderError(
                f"Qwen3-ASR 推理失败（{type(exc).__name__}）",
                reason_code="inference_failed",
            ) from exc
        if check_cancelled:
            check_cancelled()

        def output_value(name: str, default: object = None) -> object:
            if isinstance(output, dict):
                return output.get(name, default)
            return getattr(output, name, default)

        generation_tokens = 0
        language_value = output_value("language", request.language)
        if hasattr(output, "text") or isinstance(output, dict):
            # Compatibility with older mlx-audio versions and test doubles.
            segments = parse_segments(output_value("segments", ()))
            text = str(output_value("text", "") or "").strip()
            generation_tokens = int(output_value("generation_tokens", 0) or 0)
        else:
            segments = []
            committed: list[str] = []
            current: list[str] = []
            chunk_start = 0.0
            chunk_end = max(0.0, request.duration_seconds)
            token_count = 0
            try:
                for emission in output:
                    if check_cancelled:
                        check_cancelled()
                    value = str(getattr(emission, "text", "") or "")
                    if value:
                        current.append(value)
                        token_count += 1
                        if progress and token_count % 64 == 0:
                            progress(f"Qwen3-ASR 已生成 {token_count} 个解码片段")
                    language_value = getattr(emission, "language", language_value)
                    emitted_generation = int(
                        getattr(emission, "generation_tokens", 0) or 0
                    )
                    if emitted_generation:
                        generation_tokens = max(generation_tokens, emitted_generation)
                        chunk_start = float(
                            getattr(emission, "start_time", chunk_start) or 0.0
                        )
                        chunk_end = float(
                            getattr(emission, "end_time", chunk_end) or chunk_end
                        )
                        chunk_text = self._clean_stream_text("".join(current))
                        if chunk_text:
                            segments.append(AsrSegment(
                                chunk_start,
                                max(chunk_start, chunk_end),
                                chunk_text,
                            ))
                            committed.append(chunk_text)
                        current.clear()
                trailing = self._clean_stream_text("".join(current))
                if trailing:
                    segments.append(AsrSegment(
                        chunk_start,
                        max(chunk_start, chunk_end),
                        trailing,
                    ))
                    committed.append(trailing)
            except CancelledError:
                close = getattr(output, "close", None)
                if callable(close):
                    close()
                raise
            text = " ".join(committed).strip()
        if not segments and text:
            segments = [
                # Qwen supplied text but this backend version did not expose
                # aligned chunks, so retain the known source duration only.
                AsrSegment(0.0, max(0.0, request.duration_seconds), text)
            ]
        finish_reason = str(output_value("finish_reason", "") or "").strip().casefold()
        if not finish_reason:
            finish_reason = "length" if generation_tokens >= self._MAX_TOKENS else "stop"
        truncated = bool(output_value("truncated", False)) or finish_reason in {
            "length", "max_tokens", "max-tokens",
        }
        warnings: list[str] = []
        if request.word_timestamps and not any(item.words for item in segments):
            warnings.append(
                "word_timestamps_unavailable：当前 Qwen3-ASR MLX 结果未包含词级对齐；"
                "已保留句段时间戳，未伪造词级时间"
            )
        if progress:
            progress(f"Qwen3-ASR 本地转写完成，生成 {len(segments)} 个句段")
        return AsrResult(
            provider_id=self.provider_id,
            model=request.model,
            device="metal",
            compute_type=request.compute_type if request.compute_type != "auto" else "mlx",
            language=normalize_language(language_value),
            segments=segments,
            finish_reason=finish_reason,
            truncated=truncated,
            warnings=warnings,
        )

    @staticmethod
    def _clean_stream_text(value: str) -> str:
        text = value.replace("<asr_text>", "").replace("</asr_text>", "")
        text = re.sub(r"<\|[^|>]+\|>", "", text)
        return text.strip()
