from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    source = project / "src" / "media_knowledge" / "desktop" / "assets" / "ai_jingjing_mascot.png"
    destination = project / "packaging"
    png = destination / "AI-Jingjing.png"
    with Image.open(source) as original:
        image = original.convert("RGBA")
        if image.size != (1024, 1024):
            image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
        image.save(png)
    with Image.open(png) as icon:
        icon.save(
            destination / "AI-Jingjing.ico",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory(prefix="ai-jingjing-icon-") as temporary:
            iconset = Path(temporary) / "AI-Jingjing.iconset"
            iconset.mkdir()
            sizes = ((16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2))
            with Image.open(png) as icon:
                for size, scale in sizes:
                    suffix = "@2x" if scale == 2 else ""
                    icon.resize((size * scale, size * scale), Image.Resampling.LANCZOS).save(
                        iconset / f"icon_{size}x{size}{suffix}.png"
                    )
            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(destination / "AI-Jingjing.icns")],
                check=True,
            )


if __name__ == "__main__":
    main()
