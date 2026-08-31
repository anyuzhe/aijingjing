from __future__ import annotations

from ..types import CancelledError
from ...resource_scheduler import LOCAL_HEAVY_TASKS
from .base import AsrProvider, AsrProviderError, AsrRoutingError, CancellationCallback, ProgressCallback
from .types import AsrAttempt, AsrFallback, AsrResult, TranscriptionRequest


QWEN_ACCURATE_MODEL = "Qwen3-ASR-1.7B"
QWEN_FAST_MODEL = "Qwen3-ASR-0.6B"


class AsrProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, AsrProvider] = {}

    def register(self, provider: AsrProvider, *, replace: bool = False) -> None:
        provider_id = str(getattr(provider, "provider_id", "") or "").strip().casefold()
        if not provider_id:
            raise ValueError("ASR Provider 缺少 provider_id")
        if provider_id in self._providers and not replace:
            raise ValueError(f"ASR Provider 已注册：{provider_id}")
        self._providers[provider_id] = provider

    def get(self, provider_id: str) -> AsrProvider:
        normalized = str(provider_id or "").strip().casefold()
        try:
            return self._providers[normalized]
        except KeyError:
            raise AsrProviderError(
                f"未注册 ASR Provider：{normalized or 'unknown'}",
                reason_code="provider_not_registered",
            ) from None

    def contains(self, provider_id: str) -> bool:
        return str(provider_id or "").strip().casefold() in self._providers

    def providers(self) -> tuple[AsrProvider, ...]:
        return tuple(self._providers.values())

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(self._providers)


class AsrRouter:
    def __init__(self, registry: AsrProviderRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _qwen_model(request: TranscriptionRequest, default: str) -> str:
        chosen = str(request.model or "").strip()
        return chosen if "qwen3-asr" in chosen.casefold() else default

    def resolve_attempts(self, request: TranscriptionRequest) -> tuple[AsrAttempt, ...]:
        provider = request.provider
        profile = request.profile
        model = request.model or "small"
        attempts: list[AsrAttempt]

        if provider == "qwen3-mlx":
            default = QWEN_FAST_MODEL if profile == "fast-preview" else QWEN_ACCURATE_MODEL
            attempts = [
                AsrAttempt("qwen3-mlx", self._qwen_model(request, default), request.model_path, "metal", "mlx"),
                AsrAttempt("mlx-whisper", "small", request.whisper_fallback_model_path, "metal", "float16"),
                AsrAttempt("faster-whisper", "small", None, "cpu", "int8"),
            ]
        elif provider == "mlx-whisper":
            fallback_model = "small" if "turbo-q4" in model.casefold() else model
            attempts = [
                AsrAttempt("mlx-whisper", model, request.model_path, "metal", "float16"),
                AsrAttempt("faster-whisper", fallback_model, None, "cpu", "int8"),
            ]
        elif provider == "faster-whisper":
            attempts = [AsrAttempt(
                "faster-whisper",
                model,
                request.model_path,
                request.device if request.device != "auto" else "cpu",
                request.compute_type if request.compute_type != "auto" else "int8",
            )]
        elif provider:
            # Third-party providers registered by the application remain
            # selectable; unknown IDs must not silently become Whisper.
            attempts = [AsrAttempt(
                provider,
                model,
                request.model_path,
                request.device,
                request.compute_type,
            )]
        elif profile == "chinese-accuracy":
            attempts = [
                AsrAttempt("qwen3-mlx", self._qwen_model(request, QWEN_ACCURATE_MODEL), request.model_path, "metal", "mlx"),
                AsrAttempt("mlx-whisper", "small", request.whisper_fallback_model_path, "metal", "float16"),
                AsrAttempt("faster-whisper", "small", None, "cpu", "int8"),
            ]
        elif profile == "fast-preview":
            attempts = [
                AsrAttempt(
                    "qwen3-mlx", self._qwen_model(request, QWEN_FAST_MODEL), request.model_path,
                    "metal", "mlx", word_timestamps=False,
                ),
                AsrAttempt("mlx-whisper", "small", request.whisper_fallback_model_path, "metal", "float16", word_timestamps=False),
                AsrAttempt("faster-whisper", "small", None, "cpu", "int8", word_timestamps=False),
            ]
        elif profile == "compatibility":
            fallback_model = "small" if "turbo-q4" in model.casefold() else model
            attempts = [
                AsrAttempt("mlx-whisper", model, request.model_path, "metal", "float16"),
                AsrAttempt("faster-whisper", fallback_model, None, "cpu", "int8"),
            ]
        else:
            inferred = "qwen3-mlx" if "qwen3-asr" in model.casefold() else "mlx-whisper"
            return self.resolve_attempts(TranscriptionRequest(
                audio_path=request.audio_path,
                profile="custom",
                provider=inferred,
                model=model,
                model_path=request.model_path,
                whisper_fallback_model_path=request.whisper_fallback_model_path,
                language=request.language,
                context_terms=request.context_terms,
                word_timestamps=request.word_timestamps,
                allow_fallback=request.allow_fallback,
                duration_seconds=request.duration_seconds,
                device=request.device,
                compute_type=request.compute_type,
            ))
        if not request.allow_fallback:
            attempts = attempts[:1]
        return tuple(attempts)

    @staticmethod
    def _safe_unexpected_failure(provider_id: str, exc: BaseException) -> AsrProviderError:
        return AsrProviderError(
            f"{provider_id} 转写执行失败（{type(exc).__name__}）",
            reason_code="provider_runtime_failed",
        )

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress: ProgressCallback | None = None,
        check_cancelled: CancellationCallback | None = None,
    ) -> AsrResult:
        attempts = self.resolve_attempts(request)
        failures: list[AsrFallback] = []
        last_error: AsrProviderError | None = None
        for index, attempt in enumerate(attempts):
            if check_cancelled:
                check_cancelled()
            next_provider = attempts[index + 1].provider if index + 1 < len(attempts) else ""
            try:
                provider = self.registry.get(attempt.provider)
                if not provider.available():
                    raise AsrProviderError(
                        f"{attempt.provider} 当前不可用",
                        reason_code="provider_unavailable",
                    )
                resolved = request.with_attempt(
                    provider=attempt.provider,
                    model=attempt.model,
                    model_path=attempt.model_path,
                    device=attempt.device,
                    compute_type=attempt.compute_type,
                    language=attempt.language,
                    word_timestamps=attempt.word_timestamps,
                )
                if progress:
                    progress(f"正在使用 {attempt.provider} / {attempt.model} 进行本地语音识别")
                with LOCAL_HEAVY_TASKS.reserve(
                    f"asr:{attempt.provider}",
                    progress=progress,
                    check_cancelled=check_cancelled,
                ):
                    result = provider.transcribe(resolved, progress, check_cancelled)
            except CancelledError:
                raise
            except AsrProviderError as exc:
                last_error = exc
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                last_error = self._safe_unexpected_failure(attempt.provider, exc)
            else:
                result.fallback_history = [*failures, *result.fallback_history]
                return result

            if not next_provider:
                assert last_error is not None
                raise last_error
            fallback = AsrFallback(
                fallback_from=attempt.provider,
                fallback_to=next_provider,
                reason_code=last_error.reason_code if last_error else "provider_failed",
                message=str(last_error or "ASR Provider 执行失败"),
            )
            failures.append(fallback)
            if progress:
                progress(
                    f"{attempt.provider} 不可用（{fallback.reason_code}），"
                    f"已明确降级到 {next_provider}"
                )
        raise AsrRoutingError("没有可用的本地语音识别 Provider", reason_code="no_provider")
