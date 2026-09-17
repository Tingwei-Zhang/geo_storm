from examples.batch._injector import URLPrefixBlocklistRetriever


class FakeRM:
    def forward(self, query_or_queries, exclude_urls=None):
        return [
            {"url": "https://www.britannica.com/procon/pit-bulls/"},
            {"url": "https://www.britannica.com/topic/golf"},
            {"url": "https://www.reddit.com/r/pitbulls/comments/abc/"},
        ]


def test_blocks_procon_prefix_only():
    rm = URLPrefixBlocklistRetriever(FakeRM(), blocked_prefixes=["britannica.com/procon"])
    out = rm.forward("pit bulls")
    urls = [x["url"] for x in out]
    assert "https://www.reddit.com/r/pitbulls/comments/abc/" in urls
    assert "https://www.britannica.com/topic/golf" in urls
    assert all("/procon/" not in u for u in urls)
