from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from media_knowledge.desktop.controller import DesktopController
from media_knowledge.desktop.diagnostics import run_diagnostics


class TranscriptionDiagnosticsTests(unittest.TestCase):
    def test_report_exposes_machine_embedding_and_offline_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(temporary, migrate_legacy=False)
            controller.save_settings(replace(
                controller.settings,
                embedding_provider="hash",
                embedding_model="hash-384-v1",
                transcription_profile="compatibility",
                asr_provider="mlx-whisper",
                asr_model="large-v3",
                asr_model_path=None,
                auto_synthesize_notes=False,
                enable_cloud_vision=False,
                diarization_enabled=False,
            ))
            report = run_diagnostics(controller)

        self.assertIn("apple_silicon", report["machine"])
        self.assertTrue(report["embedding"]["local_ready"])
        self.assertTrue(report["offline"]["hidden_model_downloads_blocked"])
        self.assertIsInstance(report["offline"]["risks"], list)

    def test_enabled_cloud_features_are_never_reported_as_strict_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(temporary, migrate_legacy=False)
            controller.save_settings(replace(
                controller.settings,
                embedding_provider="hash",
                auto_synthesize_notes=True,
                enable_cloud_vision=False,
            ))
            report = run_diagnostics(controller)

        self.assertFalse(report["offline"]["strict_ready"])
        self.assertIn("AI 知识提炼", report["offline"]["cloud_features"])


if __name__ == "__main__":
    unittest.main()
