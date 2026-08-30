from __future__ import annotations

import importlib.util
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import queue as queue_module
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from .asr import (
    AsrProviderError,
    AsrProviderRegistry,
    AsrResult,
    AsrRouter,
    AsrSegment,
    AsrWord,
    FasterWhisperProvider,
    MlxWhisperProvider,
    Qwen3MlxProvider,
    TranscriptionRequest,
    normalize_profile,
    normalize_provider_id,
)
from .asr.providers import mlx_whisper_model
from .asr.providers._shared import resolve_local_hf_model
from .quality import evaluate_transcript_integrity
from .types import CancelledError

if TYPE_CHECKING:
    from .checkpoints import MediaCheckpointStore


# Public compatibility name for callers that persist or display word timings.
TranscriptWord = AsrWord


class TranscriptionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TranscriptionPlan:
    engine: str
    device: str
    compute_type: str
    model: str
    fallback_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    avg_logprob: float | None = None
    words: tuple[AsrWord, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class TranscriptionResult:
    plan: TranscriptionPlan
    language: str | None
    duration_seconds: float
    segments: list[TranscriptSegment]
    fallback_reasons: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    integrity: dict[str, object] = field(default_factory=dict)
    profile: str = "custom"
    provider: str | None = None
    finish_reason: str = "stop"
    truncated: bool = False
    fallback_history: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "engine": self.plan.engine,
            "device": self.plan.device,
            "compute_type": self.plan.compute_type,
            "model": self.plan.model,
            "profile": self.profile,
            "provider": self.provider or self.plan.engine,
            "language": self.language,
            "duration_seconds": round(max(0.0, self.duration_seconds), 3),
            "segment_count": len(self.segments),
            "fallback_reasons": list(dict.fromkeys([*self.plan.fallback_reasons, *self.fallback_reasons])),
            "fallback_history": list(self.fallback_history),
            "finish_reason": self.finish_reason,
            "truncated": bool(self.truncated),
            "warnings": list(dict.fromkeys(self.warnings)),
            "artifacts": dict(self.artifacts),
            "integrity": dict(self.integrity),
        }


def transcription_result_to_dict(result: TranscriptionResult) -> dict[str, object]:
    """Serialize the raw ASR fact set without relying on provider objects."""

    return {
        "format": "ai-jingjing-asr-raw-v1",
        "plan": result.plan.to_dict(),
        "language": result.language,
        "duration_seconds": result.duration_seconds,
        "segments": [item.to_dict() for item in result.segments],
        "fallback_reasons": list(result.fallback_reasons),
        "integrity": dict(result.integrity),
        "profile": result.profile,
        "provider": result.provider,
        "finish_reason": result.finish_reason,
        "truncated": result.truncated,
        "fallback_history": list(result.fallback_history),
        "warnings": list(result.warnings),
    }


def transcription_result_from_dict(value: Mapping[str, object]) -> TranscriptionResult:
    if value.get("format") != "ai-jingjing-asr-raw-v1":
        raise ValueError("ASR 检查点格式不受支持")
    raw_plan = value.get("plan")
    if not isinstance(raw_plan, Mapping):
        raise ValueError("ASR 检查点缺少执行计划")
    fallback_reasons = raw_plan.get("fallback_reasons")
    plan = TranscriptionPlan(
        engine=str(raw_plan.get("engine") or "unknown"),
        device=str(raw_plan.get("device") or "unknown"),
        compute_type=str(raw_plan.get("compute_type") or "unknown"),
        model=str(raw_plan.get("model") or "unknown"),
        fallback_reasons=tuple(
            str(item) for item in fallback_reasons
        ) if isinstance(fallback_reasons, (list, tuple)) else (),
    )
    segments: list[TranscriptSegment] = []
    raw_segments = value.get("segments")
    for item in raw_segments if isinstance(raw_segments, list) else []:
        if not isinstance(item, Mapping):
            continue
        words: list[AsrWord] = []
        raw_words = item.get("words")
        for word in raw_words if isinstance(raw_words, (list, tuple)) else ():
            if not isinstance(word, Mapping):
                continue
            words.append(AsrWord(
                float(word.get("start", 0.0)),
                float(word.get("end", 0.0)),
                str(word.get("text") or ""),
                confidence=_safe_number(word.get("confidence")),
                speaker_id=str(word.get("speaker_id") or "") or None,
            ))
        segments.append(TranscriptSegment(
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            str(item.get("text") or ""),
            confidence=_safe_number(item.get("confidence")),
            avg_logprob=_safe_number(item.get("avg_logprob")),
            words=tuple(words),
        ))
    raw_fallbacks = value.get("fallback_reasons")
    raw_history = value.get("fallback_history")
    raw_warnings = value.get("warnings")
    integrity = value.get("integrity")
    return TranscriptionResult(
        plan=plan,
        language=str(value.get("language") or "") or None,
        duration_seconds=max(0.0, float(value.get("duration_seconds") or 0.0)),
        segments=segments,
        fallback_reasons=[str(item) for item in raw_fallbacks]
        if isinstance(raw_fallbacks, list) else [],
        integrity=dict(integrity) if isinstance(integrity, Mapping) else {},
        profile=str(value.get("profile") or "custom"),
        provider=str(value.get("provider") or "") or None,
        finish_reason=str(value.get("finish_reason") or "stop"),
        truncated=bool(value.get("truncated", False)),
        fallback_history=[dict(item) for item in raw_history if isinstance(item, Mapping)]
        if isinstance(raw_history, list) else [],
        warnings=[str(item) for item in raw_warnings]
        if isinstance(raw_warnings, list) else [],
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def cuda_available() -> bool:
    if not _module_available("ctranslate2"):
        return False
    try:
        import ctranslate2  # type: ignore

        return int(ctranslate2.get_cuda_device_count()) > 0
    except (ImportError, RuntimeError, OSError, AttributeError, ValueError):
        return False


def hardware_capabilities() -> dict[str, object]:
    system = platform.system().casefold()
    machine = platform.machine().casefold()
    apple_silicon = system == "darwin" and machine in {"arm64", "aarch64"}
    return {
        "system": system,
        "machine": machine,
        "apple_silicon": apple_silicon,
        "mlx_whisper": _module_available("mlx_whisper"),
        "mlx_audio": _module_available("mlx_audio"),
        "faster_whisper": _module_available("faster_whisper"),
        "cuda": cuda_available(),
    }


def _mlx_model(model: str) -> str:
    return mlx_whisper_model(model)


def select_transcription_plan(
    model: str,
    *,
    preferred_engine: str = "auto",
    allow_cpu_fallback: bool = True,
    capabilities: dict[str, object] | None = None,
) -> TranscriptionPlan:
    """Choose an accelerated engine first and make every slow fallback explicit."""

    capabilities = dict(capabilities or hardware_capabilities())
    preferred = str(preferred_engine or "auto").strip().casefold().replace("_", "-")
    if preferred == "faster-whisper":
        preferred = "cuda" if bool(capabilities.get("cuda")) else "cpu"
    if preferred not in {"auto", "mlx", "cuda", "cpu"}:
        preferred = "auto"
    apple = bool(capabilities.get("apple_silicon"))
    mlx = bool(capabilities.get("mlx_whisper"))
    faster = bool(capabilities.get("faster_whisper"))
    cuda = bool(capabilities.get("cuda"))
    reasons: list[str] = []

    # An explicit CPU choice is not a fallback, so the fallback policy must not
    # reject it merely because the host also happens to be Apple Silicon.
    if preferred == "cpu":
        if not faster:
            raise TranscriptionUnavailable("faster-whisper 未安装，无法执行 CPU int8 转写")
        return TranscriptionPlan("faster-whisper", "cpu", "int8", model)

    if preferred in {"auto", "mlx"} and apple:
        if mlx:
            return TranscriptionPlan("mlx-whisper", "metal", "float16", _mlx_model(model))
        reasons.append("Apple Silicon 已检测到，但 mlx-whisper 未安装")
        if preferred == "mlx" and not allow_cpu_fallback:
            raise TranscriptionUnavailable(reasons[-1] + "；已禁止 CPU 慢速降级")
    elif preferred == "mlx":
        reasons.append("MLX 转写仅支持 Apple Silicon")
        if not allow_cpu_fallback:
            raise TranscriptionUnavailable(reasons[-1] + "；已禁止 CPU 慢速降级")

    if preferred in {"auto", "cuda"} and cuda:
        if not faster:
            reasons.append("检测到 NVIDIA CUDA，但 faster-whisper 未安装")
        else:
            return TranscriptionPlan("faster-whisper", "cuda", "float16", model, tuple(reasons))
    elif preferred == "cuda":
        reasons.append("未检测到可用的 NVIDIA CUDA 转写环境")
        if not allow_cpu_fallback:
            raise TranscriptionUnavailable(reasons[-1] + "；已禁止 CPU 慢速降级")

    if not faster:
        detail = "；".join(reasons + ["faster-whisper 未安装，无法执行 CPU int8 转写"])
        raise TranscriptionUnavailable(detail)
    if preferred == "auto" and not reasons:
        reasons.append("未检测到可用的 MLX 或 CUDA 转写环境")
    if not allow_cpu_fallback:
        raise TranscriptionUnavailable("；".join(reasons + ["已禁止 CPU 慢速降级"]))
    reasons.append("已明确切换到 CPU int8；长音视频处理会明显变慢")
    return TranscriptionPlan("faster-whisper", "cpu", "int8", model, tuple(reasons))


def _safe_number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _transcribe_faster_whisper(
    audio_path: Path,
    plan: TranscriptionPlan,
    check_cancelled: Callable[[], None] | None = None,
    *,
    language: str | None = None,
    context_terms: Iterable[str] = (),
    word_timestamps: bool = False,
) -> tuple[list[TranscriptSegment], str | None]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailable("faster-whisper 未安装") from exc
    constructor_kwargs: dict[str, object] = {
        "device": plan.device,
        "compute_type": plan.compute_type,
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
    model = WhisperModel(plan.model, **constructor_kwargs)
    kwargs: dict[str, object] = {"vad_filter": True, "word_timestamps": word_timestamps}
    normalized_language = str(language or "").strip().casefold()
    language_aliases = {
        "chinese": "zh", "中文": "zh", "mandarin": "zh",
        "english": "en", "英语": "en", "japanese": "ja", "日语": "ja",
        "korean": "ko", "韩语": "ko",
    }
    if normalized_language and normalized_language not in {"auto", "自动"}:
        kwargs["language"] = language_aliases.get(normalized_language, normalized_language)
    terms = [str(value).strip() for value in context_terms if str(value).strip()]
    if terms:
        context = ", ".join(terms)
        kwargs["initial_prompt"] = context
        kwargs["hotwords"] = context
    try:
        parameters = inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()
    )
    if parameters and not accepts_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    raw_segments, info = model.transcribe(str(audio_path), **kwargs)
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if check_cancelled:
            check_cancelled()
        words: list[AsrWord] = []
        for word in getattr(item, "words", ()) or ():
            start = _safe_number(getattr(word, "start", None))
            end = _safe_number(getattr(word, "end", None))
            text = str(getattr(word, "word", getattr(word, "text", "")) or "").strip()
            if start is None or end is None or not text:
                continue
            words.append(AsrWord(
                start,
                max(start, end),
                text,
                confidence=_safe_number(getattr(word, "probability", None)),
            ))
        segments.append(TranscriptSegment(
            float(item.start),
            float(item.end),
            str(item.text or "").strip(),
            avg_logprob=_safe_number(getattr(item, "avg_logprob", None)),
            words=tuple(words),
        ))
    language = str(getattr(info, "language", "") or "").strip() or None
    return segments, language


def _compact_mlx_result(result: object) -> dict[str, object]:
    """Reduce an MLX result to primitives that are safe to cross processes."""

    if not isinstance(result, dict):
        raise RuntimeError("mlx-whisper 返回了无法识别的转写结果")
    language = str(result.get("language") or "").strip() or None
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments: list[dict[str, object]] = []
    for value in raw_segments:
        if not isinstance(value, dict):
            continue
        start = _safe_number(value.get("start"))
        end = _safe_number(value.get("end"))
        if start is None or end is None:
            continue
        segment: dict[str, object] = {
            "start": start,
            "end": end,
            "text": str(value.get("text") or "").strip(),
        }
        confidence = _safe_number(value.get("confidence"))
        avg_logprob = _safe_number(value.get("avg_logprob"))
        if confidence is not None:
            segment["confidence"] = confidence
        if avg_logprob is not None:
            segment["avg_logprob"] = avg_logprob
        raw_words = value.get("words")
        words: list[dict[str, object]] = []
        if isinstance(raw_words, list):
            for raw_word in raw_words:
                if not isinstance(raw_word, dict):
                    continue
                word_start = _safe_number(raw_word.get("start"))
                word_end = _safe_number(raw_word.get("end"))
                word_text = str(raw_word.get("word") or raw_word.get("text") or "").strip()
                if word_start is None or word_end is None or not word_text:
                    continue
                word: dict[str, object] = {
                    "start": word_start,
                    "end": max(word_start, word_end),
                    "text": word_text,
                }
                probability = _safe_number(raw_word.get("probability") or raw_word.get("confidence"))
                if probability is not None:
                    word["confidence"] = probability
                words.append(word)
        if words:
            segment["words"] = words
        segments.append(segment)
    return {"language": language, "segments": segments}


def _call_mlx_transcribe(
    audio_path: str,
    model: str,
    *,
    language: str | None = None,
    context_terms: Iterable[str] = (),
    word_timestamps: bool = False,
) -> dict[str, object]:
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailable("mlx-whisper 未安装") from exc
    try:
        local_model = resolve_local_hf_model(model, provider_label="mlx-whisper")
    except AsrProviderError as exc:
        raise TranscriptionUnavailable(str(exc)) from exc
    kwargs: dict[str, object] = {
        "path_or_hf_repo": local_model,
        "word_timestamps": bool(word_timestamps),
    }
    normalized_language = str(language or "").strip().casefold()
    language_aliases = {
        "chinese": "zh", "中文": "zh", "mandarin": "zh",
        "english": "en", "英语": "en", "japanese": "ja", "日语": "ja",
        "korean": "ko", "韩语": "ko",
    }
    if normalized_language and normalized_language not in {"auto", "自动"}:
        kwargs["language"] = language_aliases.get(normalized_language, normalized_language)
    terms = [str(value).strip() for value in context_terms if str(value).strip()]
    if terms:
        kwargs["initial_prompt"] = ", ".join(terms)
    try:
        result = mlx_whisper.transcribe(audio_path, **kwargs)
    except TypeError:
        # Compatibility with earlier mlx-whisper versions whose model argument
        # is positional. Older versions cannot receive newer quality options.
        result = mlx_whisper.transcribe(audio_path, local_model)
    return _compact_mlx_result(result)


def _mlx_transcribe_worker(
    audio_path: str,
    model: str,
    result_queue: object,
    language: str | None = None,
    context_terms: tuple[str, ...] = (),
    word_timestamps: bool = False,
) -> None:
    """Run synchronous MLX inference outside the UI process.

    Only a compact primitive payload crosses the process boundary. In
    particular, exception messages and source paths are never returned.
    """

    try:
        payload = _call_mlx_transcribe(
            audio_path,
            model,
            language=language,
            context_terms=context_terms,
            word_timestamps=word_timestamps,
        )
    except BaseException as exc:  # noqa: BLE001 - subprocess must always report a safe failure
        try:
            result_queue.put(("error", type(exc).__name__))  # type: ignore[attr-defined]
        except BaseException:  # pragma: no cover - the parent also detects a dead worker
            pass
        return
    try:
        result_queue.put(("ok", payload))  # type: ignore[attr-defined]
    except BaseException:  # pragma: no cover - the parent also detects a dead worker
        pass


def _stop_mlx_worker(process: object, *, force: bool) -> None:
    """Reap a worker, escalating from terminate to kill when necessary."""

    try:
        alive = bool(process.is_alive())  # type: ignore[attr-defined]
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = False
    if force and alive:
        try:
            process.terminate()  # type: ignore[attr-defined]
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            pass
    try:
        process.join(timeout=0.75)  # type: ignore[attr-defined]
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        pass
    try:
        alive = bool(process.is_alive())  # type: ignore[attr-defined]
    except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
        alive = False
    if alive:
        try:
            process.kill()  # type: ignore[attr-defined]
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            try:
                process.terminate()  # type: ignore[attr-defined]
            except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
                pass
        try:
            process.join(timeout=0.75)  # type: ignore[attr-defined]
        except (AssertionError, AttributeError, OSError, RuntimeError, ValueError):
            pass


def _transcribe_mlx_in_worker(
    audio_path: Path,
    plan: TranscriptionPlan,
    check_cancelled: Callable[[], None],
    *,
    language: str | None = None,
    context_terms: Iterable[str] = (),
    word_timestamps: bool = False,
) -> dict[str, object]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_mlx_transcribe_worker,
        args=(
            str(audio_path), plan.model, result_queue, language,
            tuple(context_terms), bool(word_timestamps),
        ),
        daemon=True,
    )
    completed = False
    try:
        try:
            process.start()
        except (OSError, RuntimeError, ValueError):
            raise RuntimeError("MLX 转写子进程启动失败") from None
        while True:
            check_cancelled()
            try:
                message = result_queue.get(timeout=0.2)
            except queue_module.Empty:
                try:
                    alive = bool(process.is_alive())
                except (AttributeError, OSError):
                    alive = False
                if alive:
                    continue
                try:
                    message = result_queue.get_nowait()
                except queue_module.Empty:
                    raise RuntimeError("MLX 转写子进程意外结束") from None
            if not isinstance(message, tuple) or len(message) != 2:
                raise RuntimeError("MLX 转写子进程返回了无效结果")
            status, payload = message
            if status != "ok":
                # The worker intentionally sends only an exception class. Do
                # not surface it because backend errors may contain local data.
                raise RuntimeError("MLX 转写子进程执行失败")
            if not isinstance(payload, dict):
                raise RuntimeError("MLX 转写子进程返回了无效结果")
            completed = True
            return payload
    finally:
        _stop_mlx_worker(process, force=not completed)
        try:
            result_queue.close()
        except (AttributeError, OSError, ValueError):
            pass
        try:
            result_queue.cancel_join_thread()
        except (AttributeError, OSError, ValueError):
            pass


def _transcribe_mlx(
    audio_path: Path,
    plan: TranscriptionPlan,
    check_cancelled: Callable[[], None] | None = None,
    *,
    language: str | None = None,
    context_terms: Iterable[str] = (),
    word_timestamps: bool = False,
) -> tuple[list[TranscriptSegment], str | None]:
    if check_cancelled is None:
        result = _call_mlx_transcribe(
            str(audio_path), plan.model,
            language=language,
            context_terms=context_terms,
            word_timestamps=word_timestamps,
        )
    else:
        result = _transcribe_mlx_in_worker(
            audio_path, plan, check_cancelled,
            language=language,
            context_terms=context_terms,
            word_timestamps=word_timestamps,
        )
    values = result.get("segments")
    if not isinstance(values, list):
        values = []
    segments: list[TranscriptSegment] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        start = _safe_number(value.get("start"))
        end = _safe_number(value.get("end"))
        if start is None or end is None:
            continue
        parsed_words: list[AsrWord] = []
        raw_words = value.get("words")
        if isinstance(raw_words, list):
            for raw_word in raw_words:
                if not isinstance(raw_word, dict):
                    continue
                word_start = _safe_number(raw_word.get("start"))
                word_end = _safe_number(raw_word.get("end"))
                word_text = str(raw_word.get("text") or raw_word.get("word") or "").strip()
                if word_start is None or word_end is None or not word_text:
                    continue
                parsed_words.append(AsrWord(
                    word_start,
                    max(word_start, word_end),
                    word_text,
                    confidence=_safe_number(raw_word.get("confidence")),
                ))
        segments.append(
            TranscriptSegment(
                start,
                end,
                str(value.get("text") or "").strip(),
                confidence=_safe_number(value.get("confidence")),
                avg_logprob=_safe_number(value.get("avg_logprob")),
                words=tuple(parsed_words),
            )
        )
    language = str(result.get("language") or "").strip() or None
    return segments, language


def _asr_segments(values: Iterable[TranscriptSegment]) -> list[AsrSegment]:
    return [AsrSegment(
        start=item.start,
        end=item.end,
        text=item.text,
        confidence=item.confidence,
        avg_logprob=item.avg_logprob,
        words=tuple(item.words),
    ) for item in values]


def _legacy_mlx_runner(
    request: TranscriptionRequest,
    progress: Callable[[str], None] | None,
    check_cancelled: Callable[[], None] | None,
) -> AsrResult:
    del progress
    requested_reference = str(request.model_path) if request.model_path else _mlx_model(request.model)
    model_reference = resolve_local_hf_model(
        requested_reference,
        provider_label="mlx-whisper",
    )
    plan = TranscriptionPlan("mlx-whisper", "metal", "float16", model_reference)
    segments, language = _transcribe_mlx(
        request.audio_path,
        plan,
        check_cancelled,
        language=request.language,
        context_terms=request.context_terms,
        word_timestamps=request.word_timestamps,
    )
    warnings = []
    if request.word_timestamps and not any(item.words for item in segments):
        warnings.append("word_timestamps_unavailable：mlx-whisper 未返回词级时间戳")
    return AsrResult(
        provider_id="mlx-whisper",
        model=model_reference,
        device="metal",
        compute_type="float16",
        language=language,
        segments=_asr_segments(segments),
        warnings=warnings,
    )


def _legacy_faster_whisper_runner(
    request: TranscriptionRequest,
    progress: Callable[[str], None] | None,
    check_cancelled: Callable[[], None] | None,
) -> AsrResult:
    del progress
    model_reference = str(request.model_path) if request.model_path else request.model
    device = request.device if request.device != "auto" else "cpu"
    compute_type = request.compute_type if request.compute_type != "auto" else "int8"
    plan = TranscriptionPlan("faster-whisper", device, compute_type, model_reference)
    segments, language = _transcribe_faster_whisper(
        request.audio_path,
        plan,
        check_cancelled,
        language=request.language,
        context_terms=request.context_terms,
        word_timestamps=request.word_timestamps,
    )
    warnings = []
    if request.word_timestamps and not any(item.words for item in segments):
        warnings.append("word_timestamps_unavailable：faster-whisper 未返回词级时间戳")
    return AsrResult(
        provider_id="faster-whisper",
        model=model_reference,
        device=device,
        compute_type=compute_type,
        language=language,
        segments=_asr_segments(segments),
        warnings=warnings,
    )


def _transcription_registry(
    capabilities: dict[str, object],
    *,
    force_available: str | None = None,
) -> AsrProviderRegistry:
    apple = bool(capabilities.get("apple_silicon"))
    registry = AsrProviderRegistry()
    registry.register(Qwen3MlxProvider(available_override=(
        apple and bool(capabilities.get("mlx_audio"))
    )))
    registry.register(MlxWhisperProvider(
        runner=_legacy_mlx_runner,
        available_override=(
            force_available == "mlx-whisper"
            or (apple and bool(capabilities.get("mlx_whisper")))
        ),
    ))
    registry.register(FasterWhisperProvider(
        runner=_legacy_faster_whisper_runner,
        available_override=(
            force_available == "faster-whisper"
            or bool(capabilities.get("faster_whisper"))
        ),
    ))
    return registry


def _fallback_message(value: dict[str, str]) -> str:
    source = value.get("fallback_from", "ASR")
    target = value.get("fallback_to", "备用 Provider")
    reason_code = value.get("reason_code", "provider_failed")
    if target == "faster-whisper":
        target_label = "CPU int8"
    elif target == "mlx-whisper":
        target_label = "mlx-whisper small"
    else:
        target_label = target
    return f"{source} 转写失败（{reason_code}），已明确切换到 {target_label}"


def transcribe_audio(
    audio_path: str | Path,
    *,
    model: str = "small",
    profile: str = "custom",
    provider: str | None = None,
    model_path: str | Path | None = None,
    whisper_fallback_model_path: str | Path | None = None,
    language: str | None = None,
    context_terms: Iterable[str] = (),
    word_timestamps: bool = False,
    preferred_engine: str = "auto",
    allow_cpu_fallback: bool = True,
    allow_fallback: bool | None = None,
    duration_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    capabilities: dict[str, object] | None = None,
    checkpoint_store: MediaCheckpointStore | None = None,
) -> TranscriptionResult:
    """Transcribe through the pluggable ASR router.

    The historical ``model``/``preferred_engine``/``allow_cpu_fallback`` API
    remains valid. New callers can select a profile or an explicit provider and
    can pass a strictly local Qwen model directory via ``model_path``.
    """

    if check_cancelled:
        check_cancelled()
    if checkpoint_store:
        cached = checkpoint_store.read_json("asr_raw.json", "asr")
        if isinstance(cached, Mapping):
            try:
                result = transcription_result_from_dict(cached)
            except (TypeError, ValueError, OverflowError):
                pass
            else:
                if check_cancelled:
                    check_cancelled()
                if progress:
                    progress("已复用 ASR 原始结果检查点")
                return result

    target = Path(audio_path)
    resolved_profile = normalize_profile(profile)
    resolved_provider = normalize_provider_id(provider)
    effective_fallback = allow_cpu_fallback if allow_fallback is None else bool(allow_fallback)
    detected = dict(capabilities or hardware_capabilities())
    initial_reasons: tuple[str, ...] = ()
    force_available: str | None = None
    device = "auto"
    compute_type = "auto"

    # Legacy automatic selection is preserved as an adapter into the new
    # provider router. New profiles are resolved entirely by AsrRouter.
    if resolved_profile == "custom" and resolved_provider is None:
        legacy_plan = select_transcription_plan(
            model,
            preferred_engine=preferred_engine,
            allow_cpu_fallback=effective_fallback,
            capabilities=detected,
        )
        resolved_provider = legacy_plan.engine
        force_available = resolved_provider
        device = legacy_plan.device
        compute_type = legacy_plan.compute_type
        initial_reasons = legacy_plan.fallback_reasons
    elif resolved_provider == "faster-whisper":
        preferred = str(preferred_engine or "auto").strip().casefold()
        if preferred == "cuda" or (preferred == "auto" and bool(detected.get("cuda"))):
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"

    if progress:
        for reason in initial_reasons:
            progress(reason)
    raw_model_path = str(model_path or "").strip()
    raw_whisper_fallback_path = str(whisper_fallback_model_path or "").strip()
    normalized_terms: tuple[str, ...]
    if isinstance(context_terms, str):
        normalized_terms = (context_terms,)
    else:
        normalized_terms = tuple(context_terms)
    request = TranscriptionRequest(
        audio_path=target,
        profile=resolved_profile,
        provider=resolved_provider,
        model=model,
        model_path=Path(raw_model_path).expanduser() if raw_model_path else None,
        whisper_fallback_model_path=(
            Path(raw_whisper_fallback_path).expanduser()
            if raw_whisper_fallback_path else None
        ),
        language=language,
        context_terms=normalized_terms,
        word_timestamps=bool(word_timestamps),
        allow_fallback=effective_fallback,
        duration_seconds=duration_seconds,
        device=device,
        compute_type=compute_type,
    )
    router = AsrRouter(_transcription_registry(detected, force_available=force_available))
    try:
        asr_result = router.transcribe(request, progress, check_cancelled)
    except CancelledError:
        raise
    except AsrProviderError as exc:
        raise TranscriptionUnavailable(str(exc)) from exc

    fallback_history = [item.to_dict() for item in asr_result.fallback_history]
    runtime_reasons = [_fallback_message(item) for item in fallback_history]
    all_reasons = tuple(dict.fromkeys([*initial_reasons, *runtime_reasons]))
    plan = TranscriptionPlan(
        asr_result.provider_id,
        asr_result.device,
        asr_result.compute_type,
        asr_result.model,
        all_reasons,
    )
    segments = [TranscriptSegment(
        item.start,
        item.end,
        item.text,
        confidence=item.confidence,
        avg_logprob=item.avg_logprob,
        words=tuple(item.words),
    ) for item in asr_result.segments]
    integrity = evaluate_transcript_integrity(
        [segment.to_dict() for segment in segments],
        duration_seconds=duration_seconds,
    )
    result = TranscriptionResult(
        plan=plan,
        language=asr_result.language,
        duration_seconds=max(0.0, float(duration_seconds)),
        segments=segments,
        fallback_reasons=runtime_reasons,
        integrity=integrity,
        profile=resolved_profile,
        provider=asr_result.provider_id,
        finish_reason=asr_result.finish_reason,
        truncated=asr_result.truncated,
        fallback_history=fallback_history,
        warnings=list(asr_result.warnings),
    )
    if checkpoint_store:
        checkpoint_store.write_json(
            "asr_raw.json",
            "asr",
            transcription_result_to_dict(result),
            runtime={
                "asr_provider": result.provider or result.plan.engine,
                "asr_model": result.plan.model,
            },
        )
    return result


def _timestamp(value: float, *, srt: bool) -> str:
    milliseconds = max(0, round(float(value) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _clean_subtitle_text(value: str) -> str:
    return str(value or "").replace("\x00", "").replace("-->", "→").strip()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_transcript_artifacts(
    result: TranscriptionResult,
    directory: str | Path,
    basename: str,
    *,
    source_name: str,
) -> dict[str, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    clean_basename = str(basename or "").strip()
    if (
        not clean_basename
        or clean_basename in {".", ".."}
        or "\x00" in clean_basename
        or Path(clean_basename).name != clean_basename
        or "/" in clean_basename
        or "\\" in clean_basename
    ):
        raise ValueError("转写产物基础文件名不安全")
    segments = result.segments
    txt = "\n".join(
        f"[{_timestamp(item.start, srt=False)} → {_timestamp(item.end, srt=False)}] {_clean_subtitle_text(item.text)}"
        for item in segments
    )
    source_label = re.sub(r"\s+", " ", str(source_name or "未命名音视频")).strip()[:240]
    markdown_header = [
        f"# {source_label} 转写",
        "",
        f"- 引擎：{result.plan.engine}",
        f"- 设备：{result.plan.device} / {result.plan.compute_type}",
        f"- 模型：{result.plan.model}",
        f"- 语言：{result.language or 'unknown'}",
        f"- 时长：{result.duration_seconds:.3f} 秒",
        f"- 完整性：{result.integrity.get('status', 'unknown')}",
        "",
        "## 时间轴",
        "",
    ]
    markdown = "\n".join(markdown_header) + txt + ("\n" if txt else "")
    srt_blocks = [
        f"{index}\n{_timestamp(item.start, srt=True)} --> {_timestamp(item.end, srt=True)}\n{_clean_subtitle_text(item.text)}"
        for index, item in enumerate(segments, 1)
    ]
    vtt_blocks = [
        f"{_timestamp(item.start, srt=False)} --> {_timestamp(item.end, srt=False)}\n{_clean_subtitle_text(item.text)}"
        for item in segments
    ]
    payload = {
        "format": "ai-jingjing-transcript-v1",
        "source": source_label,
        **result.metadata(),
        "segments": [segment.to_dict() for segment in segments],
    }
    paths = {
        extension: target / f"{clean_basename}.{extension}"
        for extension in ("json", "md", "txt", "srt", "vtt")
    }
    _atomic_text(paths["json"], json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_text(paths["md"], markdown)
    _atomic_text(paths["txt"], txt + ("\n" if txt else ""))
    _atomic_text(paths["srt"], "\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""))
    _atomic_text(paths["vtt"], "WEBVTT\n\n" + "\n\n".join(vtt_blocks) + ("\n" if vtt_blocks else ""))
    result.artifacts = {name: str(path) for name, path in paths.items()}
    # Keep the staged JSON portable: absolute cache paths are runtime details and
    # must never be persisted into a content-addressed transcript artifact.
    portable_metadata = result.metadata()
    portable_metadata["artifacts"] = {
        name: path.name for name, path in paths.items()
    }
    payload.update(portable_metadata)
    _atomic_text(paths["json"], json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return paths
