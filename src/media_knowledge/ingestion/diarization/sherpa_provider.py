from __future__ import annotations

import array
import importlib.util
import wave
from pathlib import Path
from typing import Callable, Iterable

from .base import (
    CancellationCallback,
    DiarizationRequest,
    DiarizationResult,
    DiarizationSegment,
    DiarizationUnavailable,
    ProgressCallback,
    is_existing_local_model,
)
from .pyannote_provider import _coerce_segment


SherpaRunner = Callable[..., Iterable[object]]


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _find_models(root: Path) -> tuple[Path | None, Path | None]:
    onnx_files = sorted(path for path in root.rglob("*.onnx") if path.is_file())
    segmentation = next(
        (path for path in onnx_files if "segment" in path.name.casefold() or "pyannote" in str(path).casefold()),
        None,
    )
    embedding = next(
        (
            path
            for path in onnx_files
            if any(token in str(path).casefold() for token in ("embed", "speaker", "3dspeaker"))
            and path != segmentation
        ),
        None,
    )
    return segmentation, embedding


class SherpaOnnxProvider:
    provider_id = "sherpa-onnx"

    def __init__(self, *, runner: SherpaRunner | None = None) -> None:
        self._runner = runner

    def availability(self, request: DiarizationRequest) -> tuple[bool, str | None]:
        if not is_existing_local_model(request.model_path):
            return False, "Sherpa-ONNX 本地模型未安装；不会在转写期间自动下载"
        if self._runner is not None:
            return True, None
        if not _module_available("sherpa_onnx"):
            return False, "sherpa-onnx 运行组件未安装"
        segmentation, embedding = _find_models(request.model_path)  # type: ignore[arg-type]
        if segmentation is None or embedding is None:
            return False, "Sherpa-ONNX 本地模型目录缺少分段或说话人嵌入 ONNX 文件"
        return True, None

    def diarize(
        self,
        request: DiarizationRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> DiarizationResult:
        available, reason = self.availability(request)
        if not available:
            raise DiarizationUnavailable(reason or "Sherpa-ONNX 不可用")
        if not request.audio_path.is_file():
            raise DiarizationUnavailable(f"音频文件不存在：{request.audio_path}")
        if check_cancelled:
            check_cancelled()
        if progress:
            progress("正在运行本地 Sherpa-ONNX 说话人分段")
        runner = self._runner or self._run_local
        output = runner(
            audio_path=request.audio_path,
            model_path=request.model_path,
            num_speakers=request.exact_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            progress=progress,
            check_cancelled=check_cancelled,
        )
        if check_cancelled:
            check_cancelled()
        segments: list[DiarizationSegment] = []
        for item in output:
            if check_cancelled:
                check_cancelled()
            segments.append(_coerce_segment(item))
        result = DiarizationResult.normalized(
            provider_id=self.provider_id,
            model=str(request.model_path),
            segments=segments,
            metadata={
                "expected_speakers": request.expected_speakers,
                "min_speakers": request.min_speakers,
                "max_speakers": request.max_speakers,
                "offline": True,
            },
        )
        exact = request.exact_speakers
        if exact is not None and result.speaker_count != exact:
            result.warnings.append(
                f"用户指定 {exact} 人，但模型输出 {result.speaker_count} 个匿名说话人"
            )
        return result

    @staticmethod
    def _run_local(
        *,
        audio_path: Path,
        model_path: Path,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
        progress: ProgressCallback | None,
        check_cancelled: CancellationCallback | None,
    ) -> list[tuple[float, float, str]]:
        import sherpa_onnx  # type: ignore

        segmentation, embedding = _find_models(model_path)
        if segmentation is None or embedding is None:
            raise DiarizationUnavailable("Sherpa-ONNX 本地模型文件不完整")
        with wave.open(str(audio_path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("Sherpa-ONNX 需要 PCM S16LE 单声道工作音频")
            sample_rate = int(source.getframerate())
            samples = array.array("h", source.readframes(source.getnframes()))
        if check_cancelled:
            check_cancelled()
        floats = [sample / 32768.0 for sample in samples]
        clustering = sherpa_onnx.FastClusteringConfig(
            num_clusters=int(num_speakers) if num_speakers is not None else -1,
            threshold=0.5,
        )
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=str(segmentation.resolve())
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(embedding.resolve())
            ),
            clustering=clustering,
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        result = diarizer.process(floats, sample_rate=sample_rate)
        if check_cancelled:
            check_cancelled()
        segments = [
            (float(item.start), float(item.end), f"provider-speaker-{int(item.speaker)}")
            for item in result
        ]
        if progress:
            progress(f"Sherpa-ONNX 已生成 {len(segments)} 个说话人区间")
        return segments
