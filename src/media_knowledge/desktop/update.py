from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass(slots=True)
class UpdateReport:
    status: str
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "download_url": self.download_url,
            "notes": self.notes,
        }


def _version(value: str) -> tuple[int, ...]:
    values = []
    for part in value.strip().lstrip("v").split("."):
        digits = "".join(character for character in part if character.isdigit())
        values.append(int(digits or 0))
    return tuple(values)


def check_for_update(current_version: str, manifest_url: str | None) -> UpdateReport:
    if not manifest_url:
        return UpdateReport(
            "manual", current_version,
            notes="当前安装包未配置发布服务器；应用已具备 HTTPS 更新清单检查能力。",
        )
    if not manifest_url.startswith("https://"):
        raise ValueError("更新清单必须使用 HTTPS")
    request = urllib.request.Request(
        manifest_url,
        headers={"User-Agent": f"AI-Jingjing/{current_version}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        if response.status != 200:
            raise RuntimeError(f"更新服务器返回 HTTP {response.status}")
        if int(response.headers.get("Content-Length") or 0) > 256 * 1024:
            raise RuntimeError("更新清单异常过大")
        payload = json.loads(response.read(256 * 1024).decode("utf-8"))
    latest = str(payload.get("version") or "").strip()
    download = str(payload.get("download_url") or "").strip()
    if not latest or not download.startswith("https://"):
        raise RuntimeError("更新清单缺少有效版本号或 HTTPS 下载地址")
    status = "available" if _version(latest) > _version(current_version) else "current"
    return UpdateReport(status, current_version, latest, download, str(payload.get("notes") or ""))
