"""
Retrieval wrappers used in batch/geo experiments.

- FirstRetrievalInjector: inject manual documents once at retrieval call m/index n.
- UGCMimicRetriever: patch organic retrieval hits whose URL matches a target URL.
"""

from __future__ import annotations

from typing import Any, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from knowledge_storm.interface import Information


class FirstRetrievalInjector:
    """
    Wraps a retriever so that the provided manual documents are injected exactly once:
    on a configurable retrieval call number and at a configurable insertion position.
    Defaults preserve legacy behavior: first retrieval call and prepend at index 0.
    """

    def __init__(
        self,
        base_retriever,
        manual_documents: List[Information],
        *,
        injection_retrieval_number: int = 1,
        injection_position: int = 0,
    ):
        if injection_retrieval_number < 1:
            raise ValueError(
                f"injection_retrieval_number must be >= 1, got {injection_retrieval_number}"
            )
        if injection_position < 0:
            raise ValueError(f"injection_position must be >= 0, got {injection_position}")
        self.base_retriever = base_retriever
        self.manual_documents = manual_documents
        self.injection_retrieval_number = injection_retrieval_number
        self.injection_position = injection_position
        self.retrieval_call_count = 0
        self.injected = False
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def _manual_results(self):
        results = []
        for doc in self.manual_documents:
            results.append(
                {
                    "url": doc.url,
                    "title": doc.title,
                    "description": doc.description,
                    "snippets": doc.snippets,
                    "meta": doc.meta,
                }
            )
        return results

    def forward(self, query_or_queries, exclude_urls=None):
        results = self.base_retriever.forward(
            query_or_queries=query_or_queries, exclude_urls=exclude_urls
        )
        self.retrieval_call_count += 1
        should_inject_now = (
            (not self.injected)
            and bool(self.manual_documents)
            and self.retrieval_call_count == self.injection_retrieval_number
        )
        if should_inject_now:
            self.injected = True
            manual_results = self._manual_results()
            insert_at = min(self.injection_position, len(results))
            return results[:insert_at] + manual_results + results[insert_at:]
        return results

    __call__ = forward


def canonicalize_url(url: str) -> str:
    """Canonicalize URL for stable match (host/path/query ordering)."""
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


def extract_domain_prefix(url: str) -> str:
    """Extract a 2-level domain prefix for hierarchical UGC sites.

    Returns prefix like "reddit.com/r/Comcast_Xfinity" or empty string if
    the domain doesn't have a clear 2-level hierarchy.
    """
    parsed = urlparse(canonicalize_url(url) or url)
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    path_parts = [p for p in (parsed.path or "").split("/") if p]

    # reddit.com/r/{subreddit}
    if host in ("reddit.com", "old.reddit.com") and len(path_parts) >= 2 and path_parts[0] == "r":
        return f"reddit.com/r/{path_parts[1]}"
    # facebook.com/groups/{group}
    if host == "facebook.com" and len(path_parts) >= 2 and path_parts[0] == "groups":
        return f"facebook.com/groups/{path_parts[1]}"
    # quora.com/topic/{topic} or quora.com/profile/{user}
    if host == "quora.com" and len(path_parts) >= 2 and path_parts[0] in ("topic", "profile"):
        return f"quora.com/{path_parts[0]}/{path_parts[1]}"

    # YouTube, Medium, Instagram, TikTok, etc. — no clear 2-level hierarchy
    return ""


UGC_DOMAINS = frozenset([
    "reddit.com",
    "facebook.com",
    "youtube.com",
    "medium.com",
    "instagram.com",
    "tiktok.com",
])


class UGCBlocklistRetriever:
    """Drop retrieval results whose URL belongs to a set of UGC domains."""

    def __init__(self, base_retriever, blocked_domains: set[str] | None = None):
        self.base_retriever = base_retriever
        self.blocked = {d.lower().strip(".") for d in (blocked_domains or UGC_DOMAINS)}
        self.total_seen = 0
        self.total_blocked = 0
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def _is_blocked(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().strip(".")
        if host.startswith("www."):
            host = host[4:]
        return host in self.blocked

    def forward(self, query_or_queries, exclude_urls=None):
        results = self.base_retriever.forward(
            query_or_queries=query_or_queries, exclude_urls=exclude_urls
        )
        if not isinstance(results, list):
            return results
        filtered = []
        for item in results:
            self.total_seen += 1
            url = item.get("url", "") if isinstance(item, dict) else ""
            if url and self._is_blocked(url):
                self.total_blocked += 1
                continue
            filtered.append(item)
        return filtered

    __call__ = forward


class UGCMimicRetriever:
    """
    Patches organic retriever hits when result URL matches configured targets.

    Supports three matching modes:
      - Exact URL match against a list of target URLs
      - Domain-prefix match (e.g. "reddit.com/r/Comcast_Xfinity" matches any
        URL under that subreddit)
      - Legacy single target_url (backward compatible)

    Two patching strategies controlled by `append_mode`:
      - False (default/legacy): replace the last snippet with
        <original>\n\n<separator>\n<adversarial_text>
      - True: keep all existing snippets intact and append the adversarial
        text as an additional snippet at the end.  This simulates a single
        user-generated comment being part of the scraped page content and
        is applied every time the URL is retrieved (not just once).
    """

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

        # Build canonical target URL set
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

        # Domain prefixes (e.g. "reddit.com/r/Comcast_Xfinity")
        self.domain_prefixes: list[str] = []
        if domain_prefixes:
            self.domain_prefixes = [p.strip().rstrip("/").lower() for p in domain_prefixes if p.strip()]

        if not self.target_urls_set and not self.domain_prefixes:
            raise ValueError("UGCMimicRetriever: no target_urls or domain_prefixes provided.")
        if not self.adversarial_text:
            raise ValueError("UGCMimicRetriever adversarial_text is empty.")
        if hasattr(base_retriever, "k"):
            self.k = base_retriever.k

    def _matches(self, url: str) -> bool:
        """Check if a URL matches any target (exact URL or domain prefix)."""
        canonical = canonicalize_url(url)
        if not canonical:
            return False

        # Exact URL match
        if canonical in self.target_urls_set:
            return True

        # Domain-prefix match
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
        """Append adversarial text as if it were another comment in the thread."""
        original_clean = (original or "").strip()
        if not original_clean:
            return self.adversarial_text
        return f"{original_clean}\n\n{self.adversarial_text}"

    def _maybe_patch_one(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        current_url = (item.get("url") or "").strip()
        if not current_url:
            return item
        if not self._matches(current_url):
            return item

        patched = dict(item)
        snippets = patched.get("snippets")

        if self.append_mode:
            # Append adversarial text to the first snippet (the one the LLM
            # actually reads in Co-STORM's "brief" mode).  This simulates
            # the adversarial comment being part of the scraped page content.
            # Uses seamless join (\n\n) so it looks like any other comment.
            if isinstance(snippets, list) and snippets:
                new_snippets = list(snippets)
                first = new_snippets[0] if isinstance(new_snippets[0], str) else ""
                new_snippets[0] = self._patched_text_seamless(first)
                patched["snippets"] = new_snippets
            else:
                patched["snippets"] = [self.adversarial_text]
        else:
            # Legacy: replace last snippet
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
            query_or_queries=query_or_queries, exclude_urls=exclude_urls
        )
        if isinstance(results, list):
            return [self._maybe_patch_one(item) for item in results]
        return results

    __call__ = forward
