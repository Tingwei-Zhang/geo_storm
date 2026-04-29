import concurrent.futures
import json
import logging
import re
from typing import List, Dict
from urllib.parse import urlparse

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from trafilatura import extract

logger = logging.getLogger(__name__)


class WebPageHelper:
    """Helper class to process web pages.

    Acknowledgement: Part of the code is adapted from https://github.com/stanford-oval/WikiChat project.
    """

    def __init__(self, min_char_count: int = 150, snippet_chunk_size: int = 1000, max_thread_num: int = 10,
                 enable_arctic_shift: bool = False):
        """
        Args:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            max_thread_num: Maximum number of threads to use for concurrent requests (e.g., downloading webpages).
            enable_arctic_shift: Use Arctic Shift API for Reddit URLs to get full thread content.
        """
        self.httpx_client = httpx.Client(verify=False)
        self.min_char_count = min_char_count
        self.max_thread_num = max_thread_num
        self.enable_arctic_shift = enable_arctic_shift
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=snippet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",
                "\n",
                ".",
                "．",  # Fullwidth full stop
                "。",  # Ideographic full stop
                ",",
                "，",  # Fullwidth comma
                "、",  # Ideographic comma
                " ",
                "​",  # Zero-width space
                "",
            ],
        )

    @staticmethod
    def _is_reddit_url(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower().strip(".")
        return host in ("reddit.com", "www.reddit.com", "old.reddit.com")

    @staticmethod
    def _extract_reddit_thread_id(url: str) -> str:
        m = re.search(r"/comments/([a-z0-9]+)", url)
        return m.group(1) if m else ""

    def _fetch_reddit_text(self, url: str) -> str:
        thread_id = self._extract_reddit_thread_id(url)
        if not thread_id:
            return ""
        api_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        try:
            resp = self.httpx_client.get(
                api_url,
                params={"link_id": f"t3_{thread_id}", "limit": 100},
                timeout=10,
            )
            if resp.status_code != 200:
                return ""
            data = resp.json().get("data", [])
            bodies = [c["body"] for c in data if c.get("body") and c["body"] != "[deleted]"]
            if not bodies:
                return ""
            return "\n\n".join(bodies)
        except Exception as e:
            logger.debug("Arctic Shift fetch failed for %s: %s", url, e)
            return ""

    def download_webpage(self, url: str):
        if self.enable_arctic_shift and self._is_reddit_url(url):
            text = self._fetch_reddit_text(url)
            if text:
                return text.encode("utf-8")
        try:
            res = self.httpx_client.get(url, timeout=4)
            if res.status_code >= 400:
                res.raise_for_status()
            return res.content
        except httpx.HTTPError as exc:
            print(f"Error while requesting {exc.request.url!r} - {exc!r}")
            return None

    def urls_to_articles(self, urls: List[str]) -> Dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            htmls = list(executor.map(self.download_webpage, urls))

        articles = {}

        for h, u in zip(htmls, urls):
            if h is None:
                continue
            if self.enable_arctic_shift and self._is_reddit_url(u):
                try:
                    text = h.decode("utf-8") if isinstance(h, bytes) else h
                except Exception:
                    text = ""
                if text and len(text) > self.min_char_count:
                    articles[u] = {"text": text}
                    continue
            article_text = extract(
                h,
                include_tables=False,
                include_comments=False,
                output_format="txt",
            )
            if article_text is not None and len(article_text) > self.min_char_count:
                articles[u] = {"text": article_text}

        return articles

    def urls_to_snippets(self, urls: List[str]) -> Dict:
        articles = self.urls_to_articles(urls)
        for u in articles:
            articles[u]["snippets"] = self.text_splitter.split_text(articles[u]["text"])

        return articles
