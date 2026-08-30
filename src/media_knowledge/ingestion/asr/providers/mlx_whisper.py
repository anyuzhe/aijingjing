from __future__ import annotations

import importlib.util
import platform
from pathlib import Path
from typing import Callable

from ...types import CancelledError
from ..base import AsrProviderError, CancellationCallback, ProgressCallback
from ..types import AsrResult, TranscriptionRequest
from ._shared import (
    normalize_language,
    parse_segments,
    resolve_local_hf_model,
    value_of,
    whisper_language,
)


ProviderRunner = Callable[
    [TranscriptionRequest, ProgressCallback | None, CancellationCallback | None], AsrResult
]


def mlx_whisper_model(model: str) -> str:
    value = str(model or "small").strip()
    if "/" in value or Path(value).expanduser().exists():
        return value
    if value.endswith("-mlx"):
        return f"mlx-community/{value}"
    return f"mlx-community/whisper-{value}-mlx"


class MlxWhisperProvider:
    provider_id = "mlx-whisper"

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
        apple = platform.system().casefold() == "darwin" and platform.machine().casefold() in {
            "arm64", "aarch64",
        }
        try:
            installed = importlib.util.find_spec("mlx_whisper") is not None
        except (ImportError, AttributeError, ValueError):
            installed = False
        return apple and installed

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
            import mlx_whisper  # type: ignore
        except ImportError as exc:
            raise AsrProviderError(
                "mlx-whisper 未安装",
                reason_code="dependency_missing",
            ) from exc
        requested_model = str(request.model_path) if request.model_path else mlx_whisper_model(request.model)
        model_reference = resolve_local_hf_model(
            requested_model,
            provider_label="mlx-whisper",
        )
        kwargs: dict[str, object] = {
            "path_or_hf_repo": model_reference,
            "word_timestamps": request.word_timestamps,
        }
        language = whisper_language(request.language)
        if language:
            kwargs["language"] = language
        if request.context_terms:
            kwargs["initial_prompt"] = ", ".join(request.context_terms)
        try:
            output = mlx_whisper.transcribe(str(request.audio_path), **kwargs)
        except CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise AsrProviderError(
                f"mlx-whisper 推理失败（{type(exc).__name__}）",
                reason_code="inference_failed",
            ) from exc
        if check_cancelled:
            check_cancelled()
        segments = parse_segments(value_of(output, "segments", ()))
        warnings: list[str] = []
        if request.word_timestamps and not any(item.words for item in segments):
            warnings.append("word_timestamps_unavailable：mlx-whisper 未返回词级时间戳")
        return AsrResult(
            provider_id=self.provider_id,
            model=model_reference,
            device="metal",
            compute_type="float16",
            language=normalize_language(value_of(output, "language", request.language)),
            segments=segments,
            warnings=warnings,
        )
