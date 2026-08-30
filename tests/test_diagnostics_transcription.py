from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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

    def test_sherpa_diagnostics_resolve_the_registered_dual_onnx_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(temporary, migrate_legacy=False)
            model = Path(temporary) / "sherpa-model"
            (model / "segmentation").mkdir(parents=True)
            (model / "embedding").mkdir(parents=True)
            (model / "segmentation" / "pyannote-segmentation.onnx").write_bytes(b"seg")
            (model / "embedding" / "3dspeaker-embedding.onnx").write_bytes(b"emb")
            controller.local_models.register_path(
                "sherpa-speaker-diarization-zh", model
            )
            controller.save_settings(replace(
                controller.settings,
                embedding_provider="hash",
                auto_synthesize_notes=False,
                enable_cloud_vision=False,
                diarization_enabled=True,
                diarization_provider="sherpa",
                # Diagnostics must be able to recover the registered status
                # rather than trusting only a copied settings path.
                diarization_model_path=None,
            ))

            from media_knowledge.desktop import diagnostics

            original_available = diagnostics._available
            with patch.object(
                diagnostics,
                "_available",
                side_effect=lambda module: (
                    True if module == "sherpa_onnx" else original_available(module)
                ),
            ):
                report = run_diagnostics(controller)

        diarization = report["transcription"]["diarization"]
        self.assertTrue(diarization["ready"])
        self.assertEqual(diarization["model_path"], str(model.resolve()))
        self.assertIn("Sherpa-ONNX", diarization["reason"])


if __name__ == "__main__":
    unittest.main()
