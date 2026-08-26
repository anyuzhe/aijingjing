from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from media_knowledge.cli import run


class CLIIntegrationTests(unittest.TestCase):
    def test_multiple_files_can_be_indexed_and_searched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "cli.db"
            first = root / "first.md"
            second = root / "second.md"
            first.write_text("# Sensors\n\nLiDAR measures geometric range.", encoding="utf-8")
            second.write_text("# Sensors\n\nAn IMU measures angular velocity.", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(["--db", str(database), "index", str(first), str(second), "--collection", "Robotics"])
            self.assertEqual(code, 0)
            reports = json.loads(output.getvalue())
            self.assertEqual(len(reports), 2)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(["--db", str(database), "search", "angular velocity", "--collection", "Robotics"])
            self.assertEqual(code, 0)
            results = json.loads(output.getvalue())
            self.assertEqual(results[0]["title"], "second")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = run(["--db", str(database), "ask", "What measures angular velocity?", "--hide-evidence"])
            self.assertEqual(code, 0)
            answer = json.loads(output.getvalue())
            self.assertIn("[S1]", answer["markdown"])
            self.assertTrue(answer["citations"])
            self.assertNotIn("evidence", answer)


if __name__ == "__main__":
    unittest.main()
