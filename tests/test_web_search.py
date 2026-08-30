from __future__ import annotations

import unittest
import urllib.parse

from media_knowledge.providers import DuckDuckGoWebSearchProvider


HTML = b"""
<html><body>
  <div class="result">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fofficial%3Fx%3D1">
      Official Documentation
    </a>
    <a class="result__snippet">The official project documents the named feature.</a>
  </div>
  <div class="result">
    <a class="result__a" href="http://127.0.0.1/private">Private host</a>
    <span class="result__snippet">Must never be returned.</span>
  </div>
  <div class="result">
    <a class="result__a" href="https://research.example.org/paper">Research paper</a>
    <div class="result__snippet">Independent corroborating evidence.</div>
  </div>
</body></html>
"""


class DuckDuckGoWebSearchProviderTests(unittest.TestCase):
    def test_search_is_bounded_fixed_endpoint_and_filters_private_results(self) -> None:
        observed: dict[str, object] = {}

        def transport(request, timeout: float, max_bytes: int) -> bytes:
            observed.update(
                url=request.full_url,
                data=request.data,
                timeout=timeout,
                max_bytes=max_bytes,
            )
            return HTML

        provider = DuckDuckGoWebSearchProvider(transport=transport)
        hits = provider.search("  AGENTS.md   benchmark  ", top_k=5)

        self.assertEqual(observed["url"], "https://html.duckduckgo.com/html/")
        values = urllib.parse.parse_qs(bytes(observed["data"]).decode("utf-8"))
        self.assertEqual(values["q"], ["AGENTS.md benchmark"])
        self.assertEqual([hit.title for hit in hits], ["Official Documentation", "Research paper"])
        self.assertEqual(hits[0].url, "https://example.com/official?x=1")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_empty_query_never_calls_network(self) -> None:
        provider = DuckDuckGoWebSearchProvider(
            transport=lambda *_args: self.fail("empty query must not call transport")
        )
        self.assertEqual(provider.search("   "), [])

    def test_top_k_is_enforced(self) -> None:
        provider = DuckDuckGoWebSearchProvider(transport=lambda *_args: HTML)
        hits = provider.search("knowledge", top_k=1)
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
