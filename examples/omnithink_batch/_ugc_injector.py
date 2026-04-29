"""
UGC mimic retriever for OmniThink experiments.

Standalone version that doesn't depend on knowledge_storm.
Same logic as examples/batch/_injector.py UGCMimicRetriever.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


class UGCMimicRetriever:
    """Patches organic retrieval results when URL matches a target."""

    def __init__(
        self,
        base_retriever: Any,
        *,
        target_url: str = "",
        target_urls: list[str] | None = None,
        domain_prefixes: list[str] | None = None,
        adversarial_text: str,
        separator: str = "--- Additional comment excerpt ---",
        append_mode: bool = False,
    ):
        self.base_retriever = base_retriever
        self.adversarial_text = (adversarial_text or "").strip()
        self.separator = separator
        self.append_mode = append_mode
        self.patched_count = 0
        self.matched_urls: list[str] = []

        self.target_urls_set: set[str] = set()
        if target_urls:
            for u in target_urls:
                c = canonicalize_url(u)
                if c:
                    self.target_urls_set.add(c)
        if target_url:
            c = canonicalize_url(target_url)
            if c:
                self.target_urls_set.add(c)

        self.domain_prefixes: list[str] = []
        if domain_prefixes:
            self.domain_prefixes = [p.strip().rstrip("/").lower() for p in domain_prefixes if p.strip()]

        if not self.target_urls_set and not self.domain_prefixes:
            raise ValueError("No target_urls or domain_prefixes provided.")
        if not self.adversarial_text:
            raise ValueError("adversarial_text is empty.")
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def _matches(self, url: str) -> bool:
        canonical = canonicalize_url(url)
        if not canonical:
            return False
        if canonical in self.target_urls_set:
            return True
        if self.domain_prefixes:
            parsed = urlparse(canonical)
            host = (parsed.hostname or "").lower().strip(".")
            if host.startswith("www."):
                host = host[4:]
            url_path = f"{host}{parsed.path}".lower()
            for prefix in self.domain_prefixes:
                if url_path.startswith(prefix.lower()):
                    return True
        return False

    def _patched_text(self, original: str) -> str:
        original_clean = (original or "").strip()
        if not original_clean:
            return f"{self.separator}\n{self.adversarial_text}"
        return f"{original_clean}\n\n{self.separator}\n{self.adversarial_text}"

    def _patched_text_seamless(self, original: str) -> str:
        original_clean = (original or "").strip()
        if not original_clean:
            return self.adversarial_text
        return f"{original_clean}\n\n{self.adversarial_text}"

    def _maybe_patch_one(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        current_url = (item.get("url") or "").strip()
        if not current_url or not self._matches(current_url):
            return item

        patched = dict(item)
        snippets = patched.get("snippets")

        if self.append_mode:
            if isinstance(snippets, list) and snippets:
                new_snippets = list(snippets)
                first = new_snippets[0] if isinstance(new_snippets[0], str) else ""
                new_snippets[0] = self._patched_text_seamless(first)
                patched["snippets"] = new_snippets
            else:
                patched["snippets"] = [self.adversarial_text]
        else:
            if isinstance(snippets, list) and snippets:
                last = snippets[-1] if isinstance(snippets[-1], str) else ""
                new_snippets = list(snippets)
                new_snippets[-1] = self._patched_text(last)
                patched["snippets"] = new_snippets
            else:
                description = patched.get("description")
                patched["description"] = self._patched_text(description if isinstance(description, str) else "")

        self.patched_count += 1
        self.matched_urls.append(current_url)
        return patched

    def forward(self, query_or_queries, exclude_urls=None):
        results = self.base_retriever.forward(
            query_or_queries=query_or_queries, exclude_urls=exclude_urls or []
        )
        if isinstance(results, list):
            return [self._maybe_patch_one(item) for item in results]
        return results

    __call__ = forward
