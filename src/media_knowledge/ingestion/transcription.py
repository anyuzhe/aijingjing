from __future__ import annotations

import importlib.util
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
from typing import Callable

from .quality import evaluate_transcript_integrity
from .types import CancelledError


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

    def metadata(self) -> dict[str, object]:
        return {
            "engine": self.plan.engine,
            "device": self.plan.device,
            "compute_type": self.plan.compute_type,
            "model": self.plan.model,
            "language": self.language,
            "duration_seconds": round(max(0.0, self.duration_seconds), 3),
            "segment_count": len(self.segments),
            "fallback_reasons": list(dict.fromkeys([*self.plan.fallback_reasons, *self.fallback_reasons])),
            "artifacts": dict(self.artifacts),
            "integrity": dict(self.integrity),
        }


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
        "faster_whisper": _module_available("faster_whisper"),
        "cuda": cuda_available(),
    }


def _mlx_model(model: str) -> str:
    value = str(model or "small").strip()
    if "/" in value:
        return value
    if value.endswith("-mlx"):
        return f"mlx-community/{value}"
    return f"mlx-community/whisper-{value}-mlx"


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
) -> tuple[list[TranscriptSegment], str | None]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailable("faster-whisper 未安装") from exc
    model = WhisperModel(plan.model, device=plan.device, compute_type=plan.compute_type)
    raw_segments, info = model.transcribe(str(audio_path), vad_filter=True)
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        if check_cancelled:
            check_cancelled()
        segments.append(TranscriptSegment(
            float(item.start),
            float(item.end),
            str(item.text or "").strip(),
            avg_logprob=_safe_number(getattr(item, "avg_logprob", None)),
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
        segments.append(segment)
    return {"language": language, "segments": segments}


def _call_mlx_transcribe(audio_path: str, model: str) -> dict[str, object]:
    try:
        import mlx_whisper  # type: ignore
    except ImportError as exc:
        raise TranscriptionUnavailable("mlx-whisper 未安装") from exc
    try:
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model)
    except TypeError:
        # Compatibility with earlier mlx-whisper versions whose model argument
        # is positional.
        result = mlx_whisper.transcribe(audio_path, model)
    return _compact_mlx_result(result)


def _mlx_transcribe_worker(audio_path: str, model: str, result_queue: object) -> None:
    """Run synchronous MLX inference outside the UI process.

    Only a compact primitive payload crosses the process boundary. In
    particular, exception messages and source paths are never returned.
    """

    try:
        payload = _call_mlx_transcribe(audio_path, model)
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
) -> dict[str, object]:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_mlx_transcribe_worker,
        args=(str(audio_path), plan.model, result_queue),
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
) -> tuple[list[TranscriptSegment], str | None]:
    if check_cancelled is None:
        result = _call_mlx_transcribe(str(audio_path), plan.model)
    else:
        result = _transcribe_mlx_in_worker(audio_path, plan, check_cancelled)
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
        segments.append(
            TranscriptSegment(
                start,
                end,
                str(value.get("text") or "").strip(),
                confidence=_safe_number(value.get("confidence")),
                avg_logprob=_safe_number(value.get("avg_logprob")),
            )
        )
    language = str(result.get("language") or "").strip() or None
    return segments, language


def transcribe_audio(
    audio_path: str | Path,
    *,
    model: str,
    preferred_engine: str = "auto",
    allow_cpu_fallback: bool = True,
    duration_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    capabilities: dict[str, object] | None = None,
) -> TranscriptionResult:
    target = Path(audio_path)
    plan = select_transcription_plan(
        model,
        preferred_engine=preferred_engine,
        allow_cpu_fallback=allow_cpu_fallback,
        capabilities=capabilities,
    )
    if progress:
        for reason in plan.fallback_reasons:
            progress(reason)
        progress(f"正在使用 {plan.engine}（{plan.device}/{plan.compute_type}）进行语音识别")
    runtime_reasons: list[str] = []
    try:
        if plan.engine == "mlx-whisper":
            segments, language = _transcribe_mlx(target, plan, check_cancelled)
        else:
            segments, language = _transcribe_faster_whisper(target, plan, check_cancelled)
    except CancelledError:
        raise
    except (RuntimeError, OSError, ValueError, TranscriptionUnavailable) as exc:
        accelerated = plan.device in {"metal", "cuda"}
        capabilities = dict(capabilities or hardware_capabilities())
        if not accelerated or not allow_cpu_fallback or not bool(capabilities.get("faster_whisper")):
            raise
        reason = f"{plan.engine} {plan.device} 转写失败（{type(exc).__name__}），已明确切换到 CPU int8"
        runtime_reasons.append(reason)
        if progress:
            progress(reason + "；长音视频处理会明显变慢")
        plan = TranscriptionPlan(
            "faster-whisper",
            "cpu",
            "int8",
            model,
            tuple([*plan.fallback_reasons, reason]),
        )
        segments, language = _transcribe_faster_whisper(target, plan, check_cancelled)
    integrity = evaluate_transcript_integrity(
        [segment.to_dict() for segment in segments],
        duration_seconds=duration_seconds,
    )
    return TranscriptionResult(
        plan=plan,
        language=language,
        duration_seconds=max(0.0, float(duration_seconds)),
        segments=segments,
        fallback_reasons=runtime_reasons,
        integrity=integrity,
    )


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
