from __future__ import annotations

import importlib.util
import os
import threading
from contextlib import contextmanager
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


PyannoteRunner = Callable[..., Iterable[object]]
_OFFLINE_ENV_LOCK = threading.RLock()


@contextmanager
def _strict_offline_environment():
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    with _OFFLINE_ENV_LOCK:
        before = {key: os.environ.get(key) for key in keys}
        os.environ.update({key: "1" for key in keys})
        try:
            yield
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _coerce_segment(item: object) -> DiarizationSegment:
    if isinstance(item, DiarizationSegment):
        return item
    if isinstance(item, dict):
        return DiarizationSegment(
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            str(item.get("speaker_id") or item.get("speaker") or item.get("label") or ""),
            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
            overlap=bool(item.get("overlap", False)),
        )
    if isinstance(item, (tuple, list)) and len(item) >= 3:
        return DiarizationSegment(float(item[0]), float(item[1]), str(item[2]))
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    speaker = getattr(item, "speaker_id", None) or getattr(item, "speaker", None) or getattr(item, "label", None)
    if start is None or end is None or speaker is None:
        raise ValueError("说话人 Provider 返回了无法识别的时间区间")
    return DiarizationSegment(float(start), float(end), str(speaker))


class PyannoteProvider:
    provider_id = "pyannote"

    def __init__(self, *, runner: PyannoteRunner | None = None) -> None:
        self._runner = runner

    def availability(self, request: DiarizationRequest) -> tuple[bool, str | None]:
        if not is_existing_local_model(request.model_path):
            return False, "Pyannote 本地模型未安装；请先在模型管理器中显式安装或导入"
        if self._runner is not None:
            return True, None
        if not _module_available("pyannote.audio"):
            return False, "pyannote.audio 运行组件未安装"
        return True, None

    def diarize(
        self,
        request: DiarizationRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> DiarizationResult:
        available, reason = self.availability(request)
        if not available:
            raise DiarizationUnavailable(reason or "Pyannote 不可用")
        if not request.audio_path.is_file():
            raise DiarizationUnavailable(f"音频文件不存在：{request.audio_path}")
        if check_cancelled:
            check_cancelled()
        if progress:
            progress("正在运行本地 Pyannote 说话人分段")
        runner = self._runner or self._run_local
        exact = request.exact_speakers
        output = runner(
            audio_path=request.audio_path,
            model_path=request.model_path,
            num_speakers=exact,
            min_speakers=None if exact is not None else request.min_speakers,
            max_speakers=None if exact is not None else request.max_speakers,
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
        # The environment is forced offline before pyannote is imported or a
        # pipeline is constructed.  Only the validated local directory is used.
        with _strict_offline_environment():
            from pyannote.audio import Pipeline  # type: ignore

            if check_cancelled:
                check_cancelled()
            pipeline = Pipeline.from_pretrained(str(model_path.resolve()))
            kwargs: dict[str, int] = {}
            if num_speakers is not None:
                kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    kwargs["max_speakers"] = max_speakers
            inference = pipeline(str(audio_path.resolve()), **kwargs)
            if check_cancelled:
                check_cancelled()
            annotation = getattr(inference, "speaker_diarization", inference)
            segments: list[tuple[float, float, str]] = []
            for turn, _track, label in annotation.itertracks(yield_label=True):
                if check_cancelled:
                    check_cancelled()
                segments.append((float(turn.start), float(turn.end), str(label)))
            if progress:
                progress(f"Pyannote 已生成 {len(segments)} 个说话人区间")
            return segments
