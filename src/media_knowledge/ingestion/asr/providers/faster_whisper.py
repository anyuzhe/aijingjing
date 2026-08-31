from __future__ import annotations

import importlib.util
import inspect

from ...types import CancelledError
from ..base import AsrProviderError, CancellationCallback, ProgressCallback
from ..types import AsrResult, TranscriptionRequest
from ._shared import normalize_language, parse_segments, whisper_language
from .mlx_whisper import ProviderRunner


class FasterWhisperProvider:
    provider_id = "faster-whisper"

    def __init__(
        self,
        *,
        runner: ProviderRunner | None = None,
        available_override: bool | None = None,
    ) -> None:
        self._runner = runner
        self._available_override = available_override

    def available(self) -> bool:
        if self._available_override is not None:
            return bool(self._available_override)
        try:
            return importlib.util.find_spec("faster_whisper") is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> AsrResult:
        if self._runner is not None:
            return self._runner(request, progress, check_cancelled)
        if check_cancelled:
            check_cancelled()
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise AsrProviderError(
                "faster-whisper 未安装",
                reason_code="dependency_missing",
            ) from exc
        model_reference = str(request.model_path) if request.model_path else request.model
        try:
            constructor_kwargs: dict[str, object] = {
                "device": request.device if request.device != "auto" else "cpu",
                "compute_type": request.compute_type if request.compute_type != "auto" else "int8",
            }
            try:
                constructor_parameters = inspect.signature(WhisperModel).parameters
            except (TypeError, ValueError):
                constructor_parameters = {}
            constructor_var_kwargs = any(
                item.kind == inspect.Parameter.VAR_KEYWORD
                for item in constructor_parameters.values()
            )
            if "local_files_only" in constructor_parameters or constructor_var_kwargs:
                constructor_kwargs["local_files_only"] = True
            model = WhisperModel(model_reference, **constructor_kwargs)
            kwargs: dict[str, object] = {
                "vad_filter": True,
                "word_timestamps": request.word_timestamps,
                # Long Chinese recordings can enter a repetition loop when the
                # decoder continuously conditions on its own previous output.
                "condition_on_previous_text": False,
            }
            language = whisper_language(request.language)
            if language:
                kwargs["language"] = language
            if request.context_terms:
                context = ", ".join(request.context_terms)
                kwargs["initial_prompt"] = context
                kwargs["hotwords"] = context
            parameters = inspect.signature(model.transcribe).parameters
            accepts_kwargs = any(
                item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
            )
            if not accepts_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in parameters}
            raw_segments, info = model.transcribe(str(request.audio_path), **kwargs)
            materialized: list[object] = []
            for item in raw_segments:
                if check_cancelled:
                    check_cancelled()
                materialized.append(item)
        except CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AsrProviderError(
                f"faster-whisper 推理失败（{type(exc).__name__}）",
                reason_code="inference_failed",
            ) from exc
        segments = parse_segments(materialized)
        warnings: list[str] = []
        if request.word_timestamps and not any(item.words for item in segments):
            warnings.append("word_timestamps_unavailable：faster-whisper 未返回词级时间戳")
        return AsrResult(
            provider_id=self.provider_id,
            model=str(model_reference),
            device=request.device if request.device != "auto" else "cpu",
            compute_type=request.compute_type if request.compute_type != "auto" else "int8",
            language=normalize_language(getattr(info, "language", request.language)),
            segments=segments,
            warnings=warnings,
        )
