from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.ingestion.extractors import (
    AudioVideoExtractor,
    DirectMediaURLExtractor,
    ExtractionContext,
    PublicDownloadLimitExceeded,
    PublicDownloadProtocolRejected,
    PublicLiveStreamRejected,
    PublicPlatformVideoExtractor,
)
from media_knowledge.ingestion.quality import QualityGateError, QualityReport
from media_knowledge.ingestion.service import IngestionResult, IngestionService
from media_knowledge.ingestion.transcription import (
    TranscriptSegment,
    TranscriptionPlan,
    TranscriptionResult,
)
from media_knowledge.ingestion.types import CancelledError, CancellationToken, ExtractionResult
from media_knowledge.models import ContentSegment
from media_knowledge.product import DesktopSettings, ProductPaths
from media_knowledge.storage import KnowledgeDatabase


class PublicDownloadHardeningTests(unittest.TestCase):
    @staticmethod
    def _context(root: Path) -> ExtractionContext:
        return ExtractionContext(
            paths=ProductPaths.resolve(root).ensure(),
            settings=DesktopSettings(enable_cloud_vision=False),
            cancellation=CancellationToken(),
        )

    @staticmethod
    def _temporary_media_extractor(payload: bytes = b"quality-gated-public-media"):
        class TemporaryMediaExtractor:
            def extract(self, _url: str, context: ExtractionContext) -> ExtractionResult:
                directory = context.own_temporary_path(
                    Path(tempfile.mkdtemp(prefix="service-owned-", dir=context.paths.cache))
                )
                source = directory / "download.mp4"
                source.write_bytes(payload)
                return ExtractionResult(
                    title="质量门媒体",
                    media_type="video",
                    segments=[ContentSegment(
                        "speech-1",
                        0,
                        "speech",
                        text="这是经过转写的完整公开视频内容。",
                        location={"timestamp_start": 0, "timestamp_end": 3},
                    )],
                    source_path=source,
                    original_uri="https://youtu.be/quality-gate",
                    checksum="source-checksum",
                    metadata={
                        "platform": "youtube",
                        "content_scope": "full_media",
                        "temporary_source_owned_by": "ExtractionContext",
                        "temporary_source_identity": "stable-url",
                        "source_media_bytes": len(payload),
                    },
                )

        return TemporaryMediaExtractor()

    @staticmethod
    def _indexing_stub():
        class IndexingStub:
            def index_document(self, _document):
                return SimpleNamespace(
                    status="created",
                    document_id="document-1",
                    created_chunks=1,
                    updated_chunks=0,
                    unchanged_chunks=0,
                )

        return IndexingStub()

    @staticmethod
    def _subtitle_runner(text: str):
        def fake_run(_self, _url, options, *, download):
            info = {
                "id": "subtitle-staging",
                "title": "字幕安全课程",
                "duration": 4,
                "subtitles": {"zh": [{"ext": "vtt"}]},
            }
            if download:
                directory = Path(str(options["outtmpl"])).parent
                (directory / "subtitle-staging.zh.vtt").write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:04.000\n" + text + "\n",
                    encoding="utf-8",
                )
            return info

        return fake_run

    @staticmethod
    def _remote_response(payload: bytes):
        class Headers:
            @staticmethod
            def get_content_type() -> str:
                return "video/mp4"

            @staticmethod
            def get(name: str, default=None):
                return str(len(payload)) if name.casefold() == "content-length" else default

        response = io.BytesIO(payload)
        response.headers = Headers()
        return response

    @staticmethod
    def _media_result(path: Path) -> ExtractionResult:
        return ExtractionResult(
            title=path.stem,
            media_type="video",
            segments=[ContentSegment(
                "speech-1",
                0,
                "speech",
                text="这是远程视频中经过转写的完整内容。",
                location={"timestamp_start": 0, "timestamp_end": 2},
            )],
            source_path=path,
            checksum="placeholder",
            metadata={"transcription": {"engine": "test"}},
        )

    def test_progress_hook_enforces_reported_and_on_disk_hard_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self._context(root)
            directory = root / "download"
            directory.mkdir()
            extractor = PublicPlatformVideoExtractor()
            extractor.max_download_bytes = 8
            hook = extractor._progress_hook(context, directory)

            with self.assertRaises(PublicDownloadLimitExceeded):
                hook({"status": "downloading", "downloaded_bytes": 9})

            (directory / "fragment.part").write_bytes(b"123456789")
            with self.assertRaises(PublicDownloadLimitExceeded):
                hook({"status": "downloading", "downloaded_bytes": 1})

    def test_direct_media_download_is_uniquely_owned_and_never_overwrites_cache(self) -> None:
        payloads = [b"first-public-video", b"second-public-video"]

        def fake_extract(_self, path: Path, _context: ExtractionContext):
            return self._media_result(path)

        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch(
                "media_knowledge.ingestion.extractors.urllib.request.urlopen",
                side_effect=[self._remote_response(payload) for payload in payloads],
            ), patch.object(
                AudioVideoExtractor,
                "extract",
                autospec=True,
                side_effect=fake_extract,
            ):
                first = DirectMediaURLExtractor().extract_remote(
                    "https://cdn.example/video.mp4",
                    context,
                    original_uri="https://weixin.qq.com/sph/stable-share",
                    title="公开视频",
                )
                second = DirectMediaURLExtractor().extract_remote(
                    "https://cdn.example/video.mp4",
                    context,
                    original_uri="https://weixin.qq.com/sph/stable-share",
                    title="公开视频",
                )

            self.assertNotEqual(first.source_path.parent, second.source_path.parent)
            self.assertEqual(first.source_path.read_bytes(), payloads[0])
            self.assertEqual(second.source_path.read_bytes(), payloads[1])
            self.assertTrue(context.owns_temporary_path(first.source_path))
            self.assertTrue(context.owns_temporary_path(second.source_path))
            self.assertEqual(
                first.metadata["temporary_source_owned_by"],
                "ExtractionContext",
            )
            context.cleanup_owned_temporary_paths()
            self.assertEqual(list((context.paths.cache / "remote-media").iterdir()), [])

    def test_direct_media_probe_failure_closes_and_removes_owned_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch(
                "media_knowledge.ingestion.extractors.urllib.request.urlopen",
                return_value=self._remote_response(b"uncommitted-video"),
            ), patch.object(
                AudioVideoExtractor,
                "extract",
                side_effect=RuntimeError("probe failed"),
            ), self.assertRaisesRegex(RuntimeError, "probe failed"):
                DirectMediaURLExtractor().extract_remote(
                    "https://cdn.example/video.mp4",
                    context,
                    original_uri="https://weixin.qq.com/sph/failure",
                )

            self.assertEqual(context.owned_temporary_paths, [])
            self.assertEqual(list((context.paths.cache / "remote-media").iterdir()), [])

    def test_declared_format_sizes_are_rejected_before_media_download(self) -> None:
        extractor = PublicPlatformVideoExtractor()
        extractor.max_download_bytes = 10
        self.assertEqual(
            extractor._declared_download_size({
                "filesize_approx": 9,
                "requested_formats": [
                    {"filesize": 6},
                    {"filesize_approx": 5},
                ],
            }),
            11,
        )
        with self.assertRaises(PublicDownloadLimitExceeded):
            extractor._enforce_declared_limit({
                "requested_formats": [{"filesize": 6}, {"filesize_approx": 5}],
            })

        calls: list[bool] = []

        def fake_run(_self, _url, _options, *, download):
            calls.append(download)
            return {
                "id": "too-large",
                "requested_formats": [{"filesize": 6}, {"filesize_approx": 5}],
            }

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            PublicPlatformVideoExtractor,
            "_run_ytdlp",
            autospec=True,
            side_effect=fake_run,
        ):
            context = self._context(Path(temporary))
            with self.assertRaises(PublicDownloadLimitExceeded):
                extractor.extract("https://youtu.be/too-large", context)
            self.assertEqual(calls, [False])
            self.assertEqual(list((context.paths.cache / "public-platform").iterdir()), [])

    def test_ytdlp_wrapped_limit_and_cancellation_remain_control_errors(self) -> None:
        class WrappedYTDLP:
            control_error: BaseException

            def __init__(self, _options: dict[str, object]) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def extract_info(self, _url: str, *, download: bool):
                del download
                try:
                    raise self.control_error
                except BaseException as error:
                    raise RuntimeError("third-party wrapper") from error

        extractor = PublicPlatformVideoExtractor()
        for expected in (
            PublicDownloadLimitExceeded,
            PublicLiveStreamRejected,
            PublicDownloadProtocolRejected,
            CancelledError,
        ):
            with self.subTest(expected=expected.__name__):
                WrappedYTDLP.control_error = expected("controlled")
                with patch(
                    "media_knowledge.ingestion.extractors._youtube_dl_class",
                    return_value=WrappedYTDLP,
                ), self.assertRaises(expected):
                    extractor._run_ytdlp("https://youtu.be/public", {}, download=True)

    def test_live_probe_is_rejected_without_starting_a_download(self) -> None:
        calls: list[bool] = []

        def fake_run(_self, _url, _options, *, download):
            calls.append(download)
            return {
                "id": "live-now",
                "requested_formats": [{"live_status": "post_live", "protocol": "m3u8_native"}],
            }

        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=fake_run,
            ), self.assertRaises(PublicLiveStreamRejected):
                PublicPlatformVideoExtractor().extract("https://youtu.be/live-now", context)
            self.assertEqual(calls, [False])
            self.assertEqual(list((context.paths.cache / "public-platform").iterdir()), [])

    def test_options_force_native_fragment_downloaders_and_match_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            directory = context.paths.cache / "options"
            directory.mkdir()
            options = PublicPlatformVideoExtractor()._options(directory, context)
            self.assertNotIn("hls_prefer_native", options)
            self.assertEqual(
                options["external_downloader"],
                {"default": "native", "dash": "native"},
            )
            match_filter = options["match_filter"]
            with self.assertRaises(PublicLiveStreamRejected):
                match_filter({"requested_formats": [{"is_live": True}]}, incomplete=False)
            with self.assertRaises(PublicDownloadProtocolRejected):
                match_filter({"protocol": "rtmp"}, incomplete=False)
            for protocol in ("m3u8", "m3u8_native"):
                with self.subTest(protocol=protocol), self.assertRaisesRegex(
                    PublicDownloadProtocolRejected,
                    "HLS.*本地文件",
                ):
                    match_filter({"protocol": protocol}, incomplete=False)
            self.assertIsNone(
                match_filter(
                    {"requested_formats": [
                        {"protocol": "https"},
                        {"protocol": "http_dash_segments"},
                    ]},
                    incomplete=False,
                )
            )

    def test_hls_probe_is_rejected_before_starting_media_download(self) -> None:
        calls: list[bool] = []

        def fake_run(_self, _url, _options, *, download):
            calls.append(download)
            return {
                "id": "hls-recording",
                "title": "HLS 回放",
                "duration": 120,
                "protocol": "m3u8_native",
            }

        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=fake_run,
            ), self.assertRaisesRegex(
                PublicDownloadProtocolRejected,
                "HLS.*本地文件",
            ):
                PublicPlatformVideoExtractor().extract(
                    "https://youtu.be/hls-recording", context
                )
            self.assertEqual(calls, [False])
            self.assertEqual(list((context.paths.cache / "public-platform").iterdir()), [])

    def test_temporary_cleanup_retries_then_releases_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            directory = context.own_temporary_path(
                Path(tempfile.mkdtemp(prefix="cleanup-retry-", dir=context.paths.cache))
            )
            real_rmtree = __import__("shutil").rmtree
            attempts = 0

            def flaky_rmtree(path: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporarily locked")
                real_rmtree(path)

            with patch(
                "media_knowledge.ingestion.extractors.shutil.rmtree",
                side_effect=flaky_rmtree,
            ):
                context.cleanup_temporary_path(directory)

            self.assertEqual(attempts, 3)
            self.assertFalse(directory.exists())
            self.assertEqual(context.owned_temporary_paths, [])
            self.assertEqual(context.cleanup_failures, [])

    def test_temporary_cleanup_failure_is_recorded_and_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            directory = context.own_temporary_path(
                Path(tempfile.mkdtemp(prefix="cleanup-failure-", dir=context.paths.cache))
            )
            with patch(
                "media_knowledge.ingestion.extractors.shutil.rmtree",
                side_effect=PermissionError("still locked"),
            ), self.assertRaisesRegex(OSError, "已保留待重试记录"):
                context.cleanup_owned_temporary_paths()

            self.assertTrue(directory.exists())
            self.assertIn(directory.resolve(), context.owned_temporary_paths)
            self.assertIn(directory.resolve(), context.cleanup_failures)

            context.cleanup_owned_temporary_paths()
            self.assertFalse(directory.exists())
            self.assertEqual(context.owned_temporary_paths, [])
            self.assertEqual(context.cleanup_failures, [])

    def test_service_cleanup_diagnostic_does_not_mask_original_error(self) -> None:
        class FailingExtractor:
            def extract(self, _url: str, context: ExtractionContext) -> ExtractionResult:
                context.own_temporary_path(
                    Path(tempfile.mkdtemp(prefix="service-failure-", dir=context.paths.cache))
                )
                raise ValueError("original extraction failure")

        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(paths, settings=DesktopSettings())
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=FailingExtractor(),
            ), patch.object(
                ExtractionContext,
                "cleanup_owned_temporary_paths",
                side_effect=OSError("临时清理失败"),
            ), self.assertRaisesRegex(ValueError, "original extraction failure") as raised:
                service._ingest_one(
                    "https://example.com/failure",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://example.com/failure"],
                )
            self.assertIn("临时清理失败", " ".join(raised.exception.__notes__))

    def test_ingest_surfaces_and_durably_retries_temporary_cleanup_failure(self) -> None:
        class FailingExtractor:
            residual: Path | None = None

            def extract(self, _url: str, context: ExtractionContext) -> ExtractionResult:
                self.residual = context.own_temporary_path(
                    Path(tempfile.mkdtemp(prefix="durable-cleanup-", dir=context.paths.cache))
                )
                (self.residual / "partial.media").write_bytes(b"incomplete-public-media")
                raise ValueError("原始解析失败")

        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            extractor = FailingExtractor()
            service = IngestionService(paths, settings=DesktopSettings(enable_cloud_vision=False))
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=extractor,
            ), patch(
                "media_knowledge.ingestion.extractors.shutil.rmtree",
                side_effect=PermissionError("locked"),
            ):
                summary = service.ingest(["https://example.com/cleanup-failure"])

            result = summary.results[0]
            self.assertEqual(result.status, "failed")
            self.assertTrue(result.error.startswith("原始解析失败"))
            self.assertIn("临时清理失败", result.error)
            self.assertTrue(any("临时清理失败" in warning for warning in result.warnings))
            self.assertIsNotNone(extractor.residual)
            self.assertTrue(extractor.residual.is_dir())
            self.assertEqual(service._cleanup_registry.pending_count(), 1)
            self.assertTrue(service._cleanup_registry.registry_path.is_file())

            restarted = IngestionService(
                paths,
                settings=DesktopSettings(enable_cloud_vision=False),
            )
            self.assertFalse(extractor.residual.exists())
            self.assertEqual(restarted._cleanup_registry.pending_count(), 0)
            self.assertFalse(restarted._cleanup_registry.registry_path.exists())

    def test_cleanup_registry_never_deletes_a_directory_with_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(paths, settings=DesktopSettings(enable_cloud_vision=False))
            directory = Path(tempfile.mkdtemp(prefix="marker-check-", dir=paths.cache))
            service._cleanup_registry.register(directory)
            marker = next(directory.glob(".ai-jingjing-temporary-owner.json"))
            marker.write_text("{}\n", encoding="utf-8")

            report = service._cleanup_registry.retry_pending()

            self.assertEqual(report.removed, 0)
            self.assertEqual(report.rejected, 1)
            self.assertTrue(directory.is_dir())
            self.assertEqual(service._cleanup_registry.pending_count(), 1)

    def test_service_reports_cleanup_failure_on_successful_return(self) -> None:
        expected = IngestionResult(item="https://example.com/success", status="created")
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(paths, settings=DesktopSettings())
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=self._temporary_media_extractor(),
            ), patch.object(
                service,
                "_finish_ingestion",
                return_value=expected,
            ), patch.object(
                ExtractionContext,
                "cleanup_owned_temporary_paths",
                side_effect=OSError("临时清理失败"),
            ):
                result = service._ingest_one(
                    "https://example.com/success",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://example.com/success"],
                )
            self.assertIs(result, expected)
            self.assertEqual(result.warnings, ["导入已完成，但临时清理失败"])

    def test_cancelled_and_failed_extracts_always_remove_temporary_downloads(self) -> None:
        for error in (CancelledError("任务已取消"), RuntimeError("probe failed")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temporary:
                context = self._context(Path(temporary))
                with patch.object(
                    PublicPlatformVideoExtractor,
                    "_run_ytdlp",
                    autospec=True,
                    side_effect=error,
                ), self.assertRaises(type(error)):
                    PublicPlatformVideoExtractor().extract(
                        "https://youtu.be/cleanup", context
                    )
                cache = context.paths.cache / "public-platform"
                self.assertTrue(cache.is_dir())
                self.assertEqual(list(cache.iterdir()), [])

    def test_successful_extractor_returns_an_explicitly_owned_temporary_source(self) -> None:
        source_bytes = b"public-media-evidence"

        def fake_run(_self, _url, options, *, download):
            info = {"id": "video456", "title": "公开课程", "duration": 3}
            if download:
                directory = Path(str(options["outtmpl"])).parent
                (directory / "video456.mp4").write_bytes(source_bytes)
            return info

        extracted = ExtractionResult(
            title="video456",
            media_type="video",
            segments=[ContentSegment(
                "speech-1",
                0,
                "speech",
                text="永久证据",
                location={"timestamp_start": 0, "timestamp_end": 3},
            )],
            checksum="checksum",
            metadata={"transcription": {"engine": "faster-whisper"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=fake_run,
            ), patch.object(AudioVideoExtractor, "extract", return_value=extracted):
                result = PublicPlatformVideoExtractor().extract(
                    "https://x.com/user/status/456", context
                )

            self.assertIsNotNone(result.source_path)
            self.assertTrue(context.owns_temporary_path(result.source_path))
            self.assertTrue(Path(result.source_path).is_relative_to(context.paths.cache))
            self.assertEqual(Path(result.source_path).read_bytes(), source_bytes)
            self.assertEqual(result.metadata["temporary_source_owned_by"], "ExtractionContext")
            self.assertFalse((context.paths.assets / "public-platform" / "sources").exists())
            context.cleanup_owned_temporary_paths()
            self.assertFalse(Path(result.source_path).exists())
            self.assertEqual(list((context.paths.cache / "public-platform").iterdir()), [])

    def test_subtitle_source_and_transcript_are_context_owned_cache_staging(self) -> None:
        def fake_run(_self, _url, options, *, download):
            info = {
                "id": "subtitle123",
                "title": "字幕课程",
                "duration": 2,
                "subtitles": {"zh": [{"ext": "vtt"}]},
            }
            if download:
                directory = Path(str(options["outtmpl"])).parent
                (directory / "subtitle123.zh.vtt").write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n完整内容\n",
                    encoding="utf-8",
                )
            return info

        with tempfile.TemporaryDirectory() as temporary:
            context = self._context(Path(temporary))
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=fake_run,
            ):
                result = PublicPlatformVideoExtractor().extract(
                    "https://youtu.be/subtitle123", context
                )
            self.assertIsNotNone(result.source_path)
            self.assertTrue(Path(result.source_path).is_file())
            self.assertTrue(context.owns_temporary_path(result.source_path))
            self.assertTrue(context.owns_temporary_path(result.transcript_path))
            self.assertEqual(list(context.paths.transcripts.iterdir()), [])
            self.assertTrue(context.owned_temporary_paths)
            context.cleanup_owned_temporary_paths()
            self.assertEqual(list((context.paths.cache / "public-platform").iterdir()), [])

    def test_subtitle_quality_failure_publishes_no_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            failed_quality = QualityReport(False, 0, "F", [])
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=self._subtitle_runner("不会通过质检的字幕"),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=failed_quality,
            ), self.assertRaises(QualityGateError):
                service._ingest_one(
                    "https://youtu.be/subtitle-quality-fail",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://youtu.be/subtitle-quality-fail"],
                )

            self.assertEqual(list(paths.transcripts.iterdir()), [])
            self.assertFalse((paths.assets / "frames").exists())
            self.assertEqual(list((paths.assets / "public-platform").rglob("*")), [])

    def test_subtitle_index_failure_rolls_back_only_new_immutable_artifacts(self) -> None:
        accepted = QualityReport(True, 100, "A", [])

        class FailingIndex:
            def index_document(self, _document):
                raise RuntimeError("index failed")

        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            url = "https://youtu.be/versioned-subtitle"
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=self._subtitle_runner("第一版成功字幕"),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ):
                service._ingest_one(
                    url,
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=[url],
                )
            prior_transcripts = {
                path.resolve(): path.read_bytes()
                for path in paths.transcripts.iterdir()
                if path.is_file()
            }
            prior_sources = {
                path.resolve(): path.read_bytes()
                for path in (paths.assets / "public-platform" / "sources").iterdir()
                if path.is_file()
            }
            self.assertTrue(prior_transcripts)
            self.assertEqual(len(prior_sources), 1)

            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=self._subtitle_runner("第二版尚未提交字幕"),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ), self.assertRaisesRegex(RuntimeError, "index failed"):
                service._ingest_one(
                    url,
                    FailingIndex(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=[url],
                )

            self.assertEqual(
                {
                    path.resolve(): path.read_bytes()
                    for path in paths.transcripts.iterdir()
                    if path.is_file()
                },
                prior_transcripts,
            )
            self.assertEqual(
                {
                    path.resolve(): path.read_bytes()
                    for path in (paths.assets / "public-platform" / "sources").iterdir()
                    if path.is_file()
                },
                prior_sources,
            )

    def test_unchanged_subtitle_reimport_does_not_accumulate_orphan_artifacts(self) -> None:
        accepted = QualityReport(True, 100, "A", [])

        class UnchangedIndex:
            @staticmethod
            def index_document(_document):
                return SimpleNamespace(
                    status="unchanged",
                    document_id="document-1",
                    created_chunks=0,
                    updated_chunks=0,
                    unchanged_chunks=1,
                )

        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            url = "https://youtu.be/unchanged-subtitle"
            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=self._subtitle_runner("完全相同的稳定字幕"),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ):
                service._ingest_one(
                    url,
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=[url],
                )
            before = {
                path.resolve(): path.read_bytes()
                for path in paths.transcripts.iterdir()
                if path.is_file()
            }
            source_before = {
                path.resolve(): path.read_bytes()
                for path in (paths.assets / "public-platform" / "sources").iterdir()
                if path.is_file()
            }

            with patch.object(
                PublicPlatformVideoExtractor,
                "_run_ytdlp",
                autospec=True,
                side_effect=self._subtitle_runner("完全相同的稳定字幕"),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ):
                result = service._ingest_one(
                    url,
                    UnchangedIndex(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=[url],
                )

            self.assertEqual(result.status, "unchanged")
            self.assertEqual(
                {
                    path.resolve(): path.read_bytes()
                    for path in paths.transcripts.iterdir()
                    if path.is_file()
                },
                before,
            )
            self.assertEqual(
                {
                    path.resolve(): path.read_bytes()
                    for path in (paths.assets / "public-platform" / "sources").iterdir()
                    if path.is_file()
                },
                source_before,
            )

    def test_audio_video_quality_failure_leaves_no_transcripts_or_frames(self) -> None:
        class Vision:
            available = True

            @staticmethod
            def describe(_path, *, context):
                return f"{context}说明"

        def fake_ffmpeg(command, **_kwargs):
            output = Path(command[-1])
            if "-vn" in command:
                output.write_bytes(b"fake-wav")
            elif "-frames:v" in command:
                Path(str(output).replace("%03d", "001")).write_bytes(b"fake-frame")
            return SimpleNamespace(returncode=0)

        transcription = TranscriptionResult(
            TranscriptionPlan("test", "cpu", "int8", "tiny"),
            "zh",
            4,
            [TranscriptSegment(0, 4, "音视频暂存内容")],
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(Path(temporary) / "data").ensure()
            video = Path(temporary) / "video.mp4"
            video.write_bytes(b"video-source")
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            with patch(
                "media_knowledge.ingestion.extractors._ffmpeg_executable",
                return_value="ffmpeg",
            ), patch(
                "media_knowledge.ingestion.extractors.subprocess.run",
                side_effect=fake_ffmpeg,
            ), patch(
                "media_knowledge.ingestion.extractors._wav_duration",
                return_value=4,
            ), patch(
                "media_knowledge.ingestion.extractors.transcribe_audio",
                return_value=transcription,
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=QualityReport(False, 0, "F", []),
            ), self.assertRaises(QualityGateError):
                service._ingest_one(
                    str(video),
                    self._indexing_stub(),
                    Vision(),
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=[str(video)],
                )

            self.assertEqual(list(paths.transcripts.iterdir()), [])
            self.assertFalse((paths.assets / "frames").exists())
            self.assertEqual(list((paths.cache / "ingestion-derived").iterdir()), [])

    def test_service_quality_failure_cleans_cache_without_touching_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            existing = paths.assets / "public-platform" / "sources" / "stable-url.mp4"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"previous-success")
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            failed_quality = QualityReport(False, 0, "F", [])
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=self._temporary_media_extractor(),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=failed_quality,
            ), self.assertRaises(QualityGateError):
                service._ingest_one(
                    "https://youtu.be/quality-gate",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://youtu.be/quality-gate"],
                )
            self.assertEqual(existing.read_bytes(), b"previous-success")
            self.assertEqual(list(paths.cache.iterdir()), [])
            self.assertEqual(list(paths.archive.rglob("*.mp4")), [])

    def test_service_quality_pass_with_archive_keeps_only_archived_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=True,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            accepted = QualityReport(True, 100, "A", [])
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=self._temporary_media_extractor(),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ):
                result = service._ingest_one(
                    "https://youtu.be/quality-gate",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://youtu.be/quality-gate"],
                )
            archived_sources = list(paths.archive.glob("*/*/*/source/*.mp4"))
            self.assertEqual(len(archived_sources), 1)
            self.assertEqual(archived_sources[0].read_bytes(), b"quality-gated-public-media")
            self.assertFalse((paths.assets / "public-platform" / "sources").exists())
            self.assertEqual(list(paths.cache.iterdir()), [])
            self.assertEqual(result.archive_path, str(archived_sources[0].parent.parent))

    def test_service_quality_pass_without_archive_atomically_retains_assets_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            accepted = QualityReport(True, 100, "A", [])
            with patch(
                "media_knowledge.ingestion.service.url_extractor_for",
                return_value=self._temporary_media_extractor(),
            ), patch(
                "media_knowledge.ingestion.service.evaluate_extraction",
                return_value=accepted,
            ):
                result = service._ingest_one(
                    "https://youtu.be/quality-gate",
                    self._indexing_stub(),
                    None,
                    CancellationToken(),
                    None,
                    package_id="package-test",
                    package_members=["https://youtu.be/quality-gate"],
                )
            retained_sources = list(
                (paths.assets / "public-platform" / "sources").glob("stable-url-*.mp4")
            )
            self.assertEqual(len(retained_sources), 1)
            retained = retained_sources[0]
            self.assertEqual(retained.read_bytes(), b"quality-gated-public-media")
            self.assertEqual(list(retained.parent.glob("*.part")), [])
            self.assertEqual(list(paths.archive.rglob("*.mp4")), [])
            self.assertEqual(list(paths.cache.iterdir()), [])
            self.assertIsNone(result.archive_path)

    def test_same_url_new_content_rolls_back_without_overwriting_prior_evidence(self) -> None:
        accepted = QualityReport(True, 100, "A", [])

        class FailingIndex:
            def index_document(self, _document):
                raise RuntimeError("index failed")

        for archive_originals in (False, True):
            with self.subTest(archive_originals=archive_originals), tempfile.TemporaryDirectory() as temporary:
                paths = ProductPaths.resolve(temporary).ensure()
                service = IngestionService(
                    paths,
                    settings=DesktopSettings(
                        archive_originals=archive_originals,
                        create_source_notes=False,
                        auto_synthesize_notes=False,
                        enable_cloud_vision=False,
                    ),
                )
                url = "https://youtu.be/versioned-evidence"
                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"previous-success"),
                ):
                    first = service._ingest_one(
                        url,
                        self._indexing_stub(),
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )

                if archive_originals:
                    prior_source = next(Path(first.archive_path).glob("source/*.mp4"))
                else:
                    prior_source = next(
                        (paths.assets / "public-platform" / "sources").glob("stable-url-*.mp4")
                    )
                prior_path = prior_source.resolve()
                self.assertEqual(prior_source.read_bytes(), b"previous-success")

                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"new-uncommitted-content"),
                ), self.assertRaisesRegex(RuntimeError, "index failed"):
                    service._ingest_one(
                        url,
                        FailingIndex(),
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )

                remaining = (
                    list(paths.archive.glob("*/*/*/source/*.mp4"))
                    if archive_originals
                    else list((paths.assets / "public-platform" / "sources").glob("*.mp4"))
                )
                self.assertEqual([path.resolve() for path in remaining], [prior_path])
                self.assertEqual(prior_source.read_bytes(), b"previous-success")
                self.assertEqual(list(paths.cache.iterdir()), [])

    def test_unchanged_reimport_repairs_a_missing_database_owned_source(self) -> None:
        accepted = QualityReport(True, 100, "A", [])
        with tempfile.TemporaryDirectory() as temporary:
            paths = ProductPaths.resolve(temporary).ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=False,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            url = "https://youtu.be/repair-missing-evidence"
            with KnowledgeDatabase(paths.database) as database:
                indexing = IndexingService(
                    database,
                    HashEmbeddingProvider(dimensions=64, model="repair-test"),
                )
                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"repairable-content"),
                ):
                    first = service._ingest_one(
                        url,
                        indexing,
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )

                stored = database.get_document_by_source_id(first.source_id)
                evidence = Path(str(stored["local_path"]))
                self.assertTrue(evidence.is_file())
                evidence.unlink()
                self.assertFalse(evidence.exists())

                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"repairable-content"),
                ):
                    repaired = service._ingest_one(
                        url,
                        indexing,
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )

                self.assertEqual(repaired.status, "unchanged")
                self.assertTrue(evidence.is_file())
                self.assertEqual(evidence.read_bytes(), b"repairable-content")
                self.assertEqual(
                    database.get_document_by_source_id(first.source_id)["local_path"],
                    str(evidence),
                )

    def test_same_raw_source_with_new_parse_gets_a_versioned_consistent_bundle(self) -> None:
        accepted = QualityReport(True, 100, "A", [])

        class ParsedTextExtractor:
            def __init__(self, text: str) -> None:
                self.text = text

            def extract(self, path: Path, _context: ExtractionContext) -> ExtractionResult:
                return ExtractionResult(
                    title=path.stem,
                    media_type="markdown",
                    segments=[ContentSegment("section-1", 1, "text", text=self.text)],
                    source_path=path,
                    checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
                    metadata={"parser": "versioned-test"},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "same.md"
            source.write_text("The immutable raw source bytes.", encoding="utf-8")
            paths = ProductPaths.resolve(root / "data").ensure()
            service = IngestionService(
                paths,
                settings=DesktopSettings(
                    archive_originals=True,
                    create_source_notes=False,
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                ),
            )
            with KnowledgeDatabase(paths.database) as database:
                indexing = IndexingService(
                    database,
                    HashEmbeddingProvider(dimensions=64, model="parse-version-test"),
                )
                with patch(
                    "media_knowledge.ingestion.service.extractor_for",
                    return_value=ParsedTextExtractor("解析结果 A"),
                ), patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ):
                    first = service._ingest_one(
                        str(source),
                        indexing,
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[str(source)],
                    )
                with patch(
                    "media_knowledge.ingestion.service.extractor_for",
                    return_value=ParsedTextExtractor("解析结果 B"),
                ), patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ):
                    second = service._ingest_one(
                        str(source),
                        indexing,
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[str(source)],
                    )

                first_bundle = json.loads(
                    (Path(first.archive_path) / "bundle.json").read_text(encoding="utf-8")
                )
                second_bundle = json.loads(
                    (Path(second.archive_path) / "bundle.json").read_text(encoding="utf-8")
                )
                chunks = database.list_chunks(second.document_id)

            self.assertEqual(first.status, "created")
            self.assertEqual(second.status, "updated")
            self.assertNotEqual(first.archive_path, second.archive_path)
            self.assertNotEqual(first_bundle["parse_digest"], second_bundle["parse_digest"])
            self.assertEqual(first_bundle["segments"][0]["text"], "解析结果 A")
            self.assertEqual(second_bundle["segments"][0]["text"], "解析结果 B")
            self.assertIn("解析结果 B", chunks[0]["content"])
            self.assertNotIn("解析结果 A", chunks[0]["content"])

    def test_index_failure_never_deletes_preexisting_same_content_target(self) -> None:
        accepted = QualityReport(True, 100, "A", [])

        class FailingIndex:
            def index_document(self, _document):
                raise RuntimeError("index failed")

        for archive_originals in (False, True):
            with self.subTest(archive_originals=archive_originals), tempfile.TemporaryDirectory() as temporary:
                paths = ProductPaths.resolve(temporary).ensure()
                service = IngestionService(
                    paths,
                    settings=DesktopSettings(
                        archive_originals=archive_originals,
                        create_source_notes=False,
                        auto_synthesize_notes=False,
                        enable_cloud_vision=False,
                    ),
                )
                url = "https://youtu.be/reused-evidence"
                extractor = self._temporary_media_extractor(b"same-content")
                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=extractor,
                ):
                    first = service._ingest_one(
                        url,
                        self._indexing_stub(),
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )
                    with self.assertRaisesRegex(RuntimeError, "index failed"):
                        service._ingest_one(
                            url,
                            FailingIndex(),
                            None,
                            CancellationToken(),
                            None,
                            package_id="package-test",
                            package_members=[url],
                        )

                if archive_originals:
                    evidence = next(Path(first.archive_path).glob("source/*.mp4"))
                else:
                    evidence = next(
                        (paths.assets / "public-platform" / "sources").glob("stable-url-*.mp4")
                    )
                self.assertEqual(evidence.read_bytes(), b"same-content")

    def test_corrupted_content_addressed_target_is_rejected_without_overwrite(self) -> None:
        accepted = QualityReport(True, 100, "A", [])
        for archive_originals in (False, True):
            with self.subTest(archive_originals=archive_originals), tempfile.TemporaryDirectory() as temporary:
                paths = ProductPaths.resolve(temporary).ensure()
                service = IngestionService(
                    paths,
                    settings=DesktopSettings(
                        archive_originals=archive_originals,
                        create_source_notes=False,
                        auto_synthesize_notes=False,
                        enable_cloud_vision=False,
                    ),
                )
                url = "https://youtu.be/corrupt-evidence"
                extractor = self._temporary_media_extractor(b"original-content")
                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=extractor,
                ):
                    first = service._ingest_one(
                        url,
                        self._indexing_stub(),
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=[url],
                    )
                    if archive_originals:
                        evidence = next(Path(first.archive_path).glob("source/*.mp4"))
                    else:
                        evidence = next(
                            (paths.assets / "public-platform" / "sources").glob("stable-url-*.mp4")
                        )
                    evidence.write_bytes(b"tampered")
                    with self.assertRaisesRegex(OSError, "校验.*拒绝"):
                        service._ingest_one(
                            url,
                            self._indexing_stub(),
                            None,
                            CancellationToken(),
                            None,
                            package_id="package-test",
                            package_members=[url],
                        )

                self.assertEqual(evidence.read_bytes(), b"tampered")
                self.assertEqual(list(paths.cache.iterdir()), [])

    def test_cancellation_after_persistence_rolls_back_new_evidence(self) -> None:
        accepted = QualityReport(True, 100, "A", [])
        for archive_originals in (False, True):
            with self.subTest(archive_originals=archive_originals), tempfile.TemporaryDirectory() as temporary:
                paths = ProductPaths.resolve(temporary).ensure()
                service = IngestionService(
                    paths,
                    settings=DesktopSettings(
                        archive_originals=archive_originals,
                        create_source_notes=False,
                        auto_synthesize_notes=False,
                        enable_cloud_vision=False,
                    ),
                )
                token = CancellationToken()
                method_name = "_archive" if archive_originals else "_retain_unarchived_source"
                original_persist = getattr(service, method_name)

                def persist_then_cancel(*args, **kwargs):
                    persisted = original_persist(*args, **kwargs)
                    token.cancel()
                    return persisted

                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"cancelled-before-index"),
                ), patch.object(
                    service,
                    method_name,
                    side_effect=persist_then_cancel,
                ), self.assertRaises(CancelledError):
                    service._ingest_one(
                        "https://youtu.be/cancel-before-index",
                        self._indexing_stub(),
                        None,
                        token,
                        None,
                        package_id="package-test",
                        package_members=["https://youtu.be/cancel-before-index"],
                    )

                self.assertEqual(list(paths.archive.glob("*/*/*/source/*.mp4")), [])
                self.assertEqual(
                    list((paths.assets / "public-platform" / "sources").glob("*.mp4")),
                    [],
                )
                self.assertEqual(list(paths.cache.iterdir()), [])

    def test_successful_index_commits_evidence_before_source_note_failure(self) -> None:
        accepted = QualityReport(True, 100, "A", [])
        for archive_originals in (False, True):
            with self.subTest(archive_originals=archive_originals), tempfile.TemporaryDirectory() as temporary:
                paths = ProductPaths.resolve(temporary).ensure()
                service = IngestionService(
                    paths,
                    settings=DesktopSettings(
                        archive_originals=archive_originals,
                        create_source_notes=True,
                        auto_synthesize_notes=False,
                        enable_cloud_vision=False,
                    ),
                )
                with patch(
                    "media_knowledge.ingestion.service.evaluate_extraction",
                    return_value=accepted,
                ), patch(
                    "media_knowledge.ingestion.service.url_extractor_for",
                    return_value=self._temporary_media_extractor(b"indexed-evidence"),
                ), patch.object(
                    service,
                    "_write_source_note",
                    side_effect=OSError("note write failed"),
                ), self.assertRaisesRegex(OSError, "note write failed"):
                    service._ingest_one(
                        "https://youtu.be/index-committed",
                        self._indexing_stub(),
                        None,
                        CancellationToken(),
                        None,
                        package_id="package-test",
                        package_members=["https://youtu.be/index-committed"],
                    )

                retained = (
                    list(paths.archive.glob("*/*/*/source/*.mp4"))
                    if archive_originals
                    else list((paths.assets / "public-platform" / "sources").glob("*.mp4"))
                )
                self.assertEqual(len(retained), 1)
                self.assertEqual(retained[0].read_bytes(), b"indexed-evidence")
                self.assertEqual(list(paths.cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
