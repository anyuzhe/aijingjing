from __future__ import annotations

import html
import ipaddress
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable


@dataclass(slots=True)
class WebSearchHit:
    title: str
    content: str
    url: str
    score: float = 0.0
    published_at: str | None = None


class WebSearchProvider(ABC):
    name: str

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        raise NotImplementedError


class DisabledWebSearchProvider(WebSearchProvider):
    name = "disabled"

    @property
    def available(self) -> bool:
        return False

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        return []


WebTransport = Callable[[urllib.request.Request, float, int], bytes]


def _default_transport(
    request: urllib.request.Request,
    timeout: float,
    max_bytes: int,
) -> bytes:
    """Read a bounded HTTPS response without retaining cookies or credentials."""

    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = str(response.headers.get("Content-Type") or "").casefold()
        if "text/html" not in content_type:
            raise RuntimeError("外部检索返回了非 HTML 内容")
        declared = response.headers.get("Content-Length")
        if declared:
            try:
                if int(declared) > max_bytes:
                    raise RuntimeError("外部检索响应超过安全大小限制")
            except ValueError:
                pass
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise RuntimeError("外部检索响应超过安全大小限制")
        return body


def _public_result_url(value: str) -> str | None:
    """Return an HTTP(S) search-result URL while rejecting local destinations."""

    candidate = html.unescape(str(value or "").strip())
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    parsed = urllib.parse.urlsplit(candidate)
    if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/l/":
        redirect = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        candidate = urllib.parse.unquote(redirect)
        parsed = urllib.parse.urlsplit(candidate)
    if parsed.scheme.casefold() not in {"http", "https"}:
        return None
    if not parsed.hostname or parsed.username or parsed.password:
        return None
    host = parsed.hostname.casefold().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local")):
        return None
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return None
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str, str]] = []
        self._url = ""
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._capture: str | None = None

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        raw = next((value for key, value in attrs if key == "class"), "") or ""
        return {item for item in raw.split() if item}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = self._classes(attrs)
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._url = next((value for key, value in attrs if key == "href"), "") or ""
            self._capture = "title"
        elif "result__snippet" in classes and self._url:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div", "span"}:
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self._title.append(data)
        elif self._capture == "snippet":
            self._snippet.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._url:
            title = " ".join("".join(self._title).split())
            snippet = " ".join("".join(self._snippet).split())
            self.results.append((title, snippet, self._url))
        self._url = ""
        self._title = []
        self._snippet = []
        self._capture = None


class DuckDuckGoWebSearchProvider(WebSearchProvider):
    """Optional external evidence search with bounded, fixed-endpoint requests.

    Search snippets remain untrusted evidence.  Consumers must retain the URL and
    may use them to corroborate a correction, but must never treat page text as
    executable instructions.
    """

    name = "duckduckgo-html"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        *,
        timeout: float = 15.0,
        max_response_bytes: int = 2 * 1024 * 1024,
        transport: WebTransport | None = None,
    ) -> None:
        self.timeout = max(1.0, min(60.0, float(timeout)))
        self.max_response_bytes = max(64 * 1024, min(8 * 1024 * 1024, int(max_response_bytes)))
        self._transport = transport or _default_transport

    @property
    def available(self) -> bool:
        return True

    def search(self, query: str, top_k: int = 5) -> list[WebSearchHit]:
        normalized = " ".join(str(query or "").split())[:512]
        if not normalized:
            return []
        limit = max(1, min(10, int(top_k)))
        request = urllib.request.Request(
            self.endpoint,
            data=urllib.parse.urlencode({"q": normalized, "kl": "cn-zh"}).encode("utf-8"),
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "AI-Jingjing/2.4 (+local knowledge evidence verification)",
            },
            method="POST",
        )
        body = self._transport(request, self.timeout, self.max_response_bytes)
        parser = _DuckDuckGoHTMLParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        parser.close()
        hits: list[WebSearchHit] = []
        seen: set[str] = set()
        for title, snippet, raw_url in parser.results:
            url = _public_result_url(raw_url)
            if not url or url in seen or not title:
                continue
            seen.add(url)
            hits.append(
                WebSearchHit(
                    title=title[:300],
                    content=snippet[:1500],
                    url=url,
                    score=round(1.0 / (len(hits) + 1), 6),
                )
            )
            if len(hits) >= limit:
                break
        return hits


__all__ = [
    "DisabledWebSearchProvider",
    "DuckDuckGoWebSearchProvider",
    "WebSearchHit",
    "WebSearchProvider",
]
