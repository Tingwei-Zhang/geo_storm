import logging
import os
from typing import List, Dict, Any, Optional, Union, Callable
import dspy
import requests
import re
import uuid
import json
import random
import time
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
from sentence_transformers import SentenceTransformer

from src.utils.WebPageHelper import WebPageHelper


def clean_text(res):
    pattern = r'\[.*?\]\(.*?\)'
    result = re.sub(pattern, '', res)
    url_pattern = pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    result = re.sub(url_pattern, '', result)
    result = re.sub(r"\n\n+", "\n", result)
    return result

class GoogleSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):

        super().__init__(k=k)
        # key = os.environ.get('SEARCHKEY', 'default_value')
        key = "19WaNVGhRjcjYcKuOV96w"
        self.header = {
            "Content-Type": "application/json",
            "Accept-Encoding": "utf-8",
            "Authorization": f"Bearer lm-/{key}== ",
        }

        self.template = {
            "rid": str(uuid.uuid4()),
            "scene": "dolphin_search_bing_nlp",
            "uq": "",
            "debug": True,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "multiSearch": False,
                "qpMultiQuery": False,
                "qpMultiQueryHistory": [],
                "qpSpellcheck": False,
                "qpEmbedding": False,
                "knnWithScript": False,
                "qpTermsWeight": False,
                "pluginServiceConfig": {"qp": "mvp_search_qp_qwen"},  # v3 rewrite
            },
            "headers": {"__d_head_qto": 5000},
        }
        
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):

        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        MAX_RETRIES = 30
        for query in queries:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    self.template["uq"] = query

                    response = requests.post(
                        "https://nlp-cn-beijing.aliyuncs.com/gw/v1/api/msearch-sp/qwen-search",
                        data=json.dumps(self.template),
                        headers=self.header,
                    )
                    response = json.loads(response.text)
                    search_results = response['data']['docs']
                    for result in search_results:
                        url_to_results[result['url']] = {
                            'url': result['url'],
                            'title': result['title'],
                            'description': result.get('snippet', '')
                        }
                except Exception as e:
                    retries += 1
                    RETRY_DELAY = random.uniform(0, 10)
                    logging.error(f"Error occurred when searching query {query}: {e}")
                    if retries < MAX_RETRIES:
                        logging.info(f"Retrying ({retries}/{MAX_RETRIES}) after {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                    else:
                        logging.error(f"Max retries reached for query {query}. Skipping.")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)

        print(f'lengt of collected_results :{len(collected_results)}')
        return collected_results


class SerperSearch(dspy.Retrieve):
    """Search using Serper (Google Search API via serper.dev). Requires SERPER_API_KEY env var.

    Same interface as GoogleSearch: returns list of dicts with 'url', 'title', 'description', 'snippets'.
    Uses WebPageHelper for full-page snippet extraction (matching GoogleSearch behaviour).
    """

    def __init__(
        self,
        serper_search_api_key=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
        enable_arctic_shift: bool = False,
        merge_snippets: bool = False,
    ):
        super().__init__(k=k)
        self.serper_search_api_key = serper_search_api_key or os.environ.get("SERPER_API_KEY")
        if not self.serper_search_api_key:
            raise RuntimeError(
                "You must supply serper_search_api_key or set the SERPER_API_KEY environment variable"
            )

        self.base_url = "https://google.serper.dev"
        self.query_params = {"num": k, "autocorrect": True, "page": 1}
        self.merge_snippets = merge_snippets

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
            enable_arctic_shift=enable_arctic_shift,
        )
        self.usage = 0

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SerperSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search using Serper API for self.k top results, then fetch full page snippets."""
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
        }

        for query in queries:
            try:
                params = {**self.query_params, "q": query, "type": "search"}
                response = requests.post(
                    f"{self.base_url}/search",
                    headers=headers,
                    json=params,
                )
                result = response.json()

                for item in result.get("organic", []):
                    url = item.get("link", "")
                    if (
                        url
                        and self.is_valid_source(url)
                        and url not in exclude_urls
                    ):
                        url_to_results[url] = {
                            "title": item.get("title", ""),
                            "url": url,
                            "description": item.get("snippet", ""),
                        }
            except Exception as e:
                logging.error(f"Error occurred while searching query {query}: {e}")

        # Fetch full page content via WebPageHelper (same as GoogleSearch)
        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in url_to_results:
            r = url_to_results[url].copy()
            if url in valid_url_to_snippets:
                page_snippets = valid_url_to_snippets[url]["snippets"]
                desc = r.get("description", "") or ""
                if self.merge_snippets and page_snippets:
                    merged = "\n\n".join(
                        s for s in [desc] + page_snippets if s
                    )
                    r["snippets"] = [merged]
                else:
                    r["snippets"] = page_snippets
            else:
                desc = r.get("description", "") or ""
                r["snippets"] = [desc] if desc.strip() else []
            collected_results.append(r)

        return collected_results


class GoogleSearch(dspy.Retrieve):
    """Search using Google Custom Search API. Requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID."""

    def __init__(
        self,
        google_search_api_key=None,
        google_cse_id=None,
        k=3,
        is_valid_source: Callable = None,
        min_char_count: int = 150,
        snippet_chunk_size: int = 1000,
        webpage_helper_max_threads=10,
    ):
        super().__init__(k=k)
        try:
            from googleapiclient.discovery import build
        except ImportError as err:
            raise ImportError(
                "GoogleSearch requires `pip install google-api-python-client`."
            ) from err
        if not google_search_api_key and not os.environ.get("GOOGLE_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply google_search_api_key or set the GOOGLE_SEARCH_API_KEY environment variable"
            )
        if not google_cse_id and not os.environ.get("GOOGLE_CSE_ID"):
            raise RuntimeError(
                "You must supply google_cse_id or set the GOOGLE_CSE_ID environment variable"
            )

        self.google_search_api_key = (
            google_search_api_key or os.environ["GOOGLE_SEARCH_API_KEY"]
        )
        self.google_cse_id = google_cse_id or os.environ["GOOGLE_CSE_ID"]

        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

        self.service = build(
            "customsearch", "v1", developerKey=self.google_search_api_key
        )
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads,
        )
        self.usage = 0

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"GoogleSearch": usage}

    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []
    ):
        """Search using Google Custom Search API for self.k top results for query or queries."""
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        for query in queries:
            try:
                response = (
                    self.service.cse()
                    .list(
                        q=query,
                        cx=self.google_cse_id,
                        num=self.k,
                    )
                    .execute()
                )

                for item in response.get("items", []):
                    if (
                        self.is_valid_source(item["link"])
                        and item["link"] not in exclude_urls
                    ):
                        url_to_results[item["link"]] = {
                            "title": item["title"],
                            "url": item["link"],
                            "description": item.get("snippet", ""),
                        }

            except Exception as e:
                logging.error(f"Error occurred while searching query {query}: {e}")

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(
            list(url_to_results.keys())
        )
        collected_results = []
        for url in url_to_results:
            r = url_to_results[url].copy()
            if url in valid_url_to_snippets:
                r["snippets"] = valid_url_to_snippets[url]["snippets"]
            else:
                desc = r.get("description", "") or ""
                r["snippets"] = [desc] if desc.strip() else []
            collected_results.append(r)

        return collected_results

    

class BingSearchAli(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en-US', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("SEARCH_ALI_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_api_key or set environment variable SEARCH_ALI_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["SEARCH_ALI_API_KEY"]
        self.endpoint = "https://idealab.alibaba-inc.com/api/v1/search/search"
        self.count = k
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        payload_template = {
            "query": "pleaceholder",
            "num": self.count,
            "extendParams": {
                "country": "US",
                "locale": "en-US",
                "location": "United States",
                "page": 2
            },
            "platformInput": {
                "model": "google-search",
                "instanceVersion": "S1"
            }
        }
        header = {"X-AK": self.bing_api_key, "Content-Type": "application/json"}

        for query in queries:
            try:
                payload_template["query"] = query
                response = requests.post(
                    self.endpoint,
                    headers=header,
                    json=payload_template,
                ).json()
                search_results = response['data']['originalOutput']['webPages']['value']

                for result in search_results:
                    url_to_results[result['url']] = {
                        'url': result['url'],
                        'title': result['name'],
                        'description': result.get('snippet', '')
                    }
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


class BingSearch(dspy.Retrieve):
    def __init__(self, bing_search_api_key=None, k=3, is_valid_source: Callable = None,
                 min_char_count: int = 150, snippet_chunk_size: int = 1000, webpage_helper_max_threads=10,
                 mkt='en-US', language='en', **kwargs):
        """
        Params:
            min_char_count: Minimum character count for the article to be considered valid.
            snippet_chunk_size: Maximum character count for each snippet.
            webpage_helper_max_threads: Maximum number of threads to use for webpage helper.
            mkt, language, **kwargs: Bing search API parameters.
            - Reference: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY")
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.params = {
            'mkt': mkt,
            "setLang": language,
            "count": k,
            **kwargs
        }
        self.webpage_helper = WebPageHelper(
            min_char_count=min_char_count,
            snippet_chunk_size=snippet_chunk_size,
            max_thread_num=webpage_helper_max_threads
        )
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {'BingSearch': usage}

    def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []):
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        url_to_results = {}

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key , "Content-Type": "application/json" }

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint,
                    headers=headers,
                    params={**self.params, 'q': query}
                ).json()

                for d in results['webPages']['value']:
                    if self.is_valid_source(d['url']) and d['url'] not in exclude_urls:
                        url_to_results[d['url']] = {'url': d['url'], 'title': d['name'], 'description': d['snippet']}
            except Exception as e:
                logging.error(f'Error occurs when searching query {query}: {e}')

        valid_url_to_snippets = self.webpage_helper.urls_to_snippets(list(url_to_results.keys()))
        collected_results = []
        for url in valid_url_to_snippets:
            r = url_to_results[url]
            r['snippets'] = valid_url_to_snippets[url]['snippets']
            collected_results.append(r)
        return collected_results


if faiss is not None:
    class OfflineRAGFlow:
        def __init__(self,
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     chunk_size: int = 800,
                     overlap: int = 120,
                     k: int = 5):
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            self.chunk_size = chunk_size
            self.overlap = overlap
            self.k = k

            self.docs: Dict[str, Dict[str, Any]] = {}
            self.keys: List[tuple] = []
            self.embeddings: Optional[np.ndarray] = None
            self.index: Optional[faiss.IndexFlatIP] = None

        # -----------------------------
        # Helper functions
        # -----------------------------
        def _clean_text(self, text: str) -> str:
            text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # remove markdown links
            text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", '', text)  # remove urls
            text = re.sub(r"\n\n+", "\n", text)
            return text.strip()

        def _chunk_text(self, text: str) -> List[str]:
            text = self._clean_text(text)
            n = len(text)
            if n == 0:
                return []
            chunks = []
            i = 0
            while i < n:
                j = min(n, i + self.chunk_size)
                slice_ = text[i:j]
                k = slice_.rfind("。")
                if k == -1:
                    k = slice_.rfind(".")
                if k != -1 and (i + k + 1 - i) > self.chunk_size * 0.6:
                    j = i + k + 1
                chunks.append(text[i:j].strip())
                if j >= n:
                    break
                i = max(j - self.overlap, i + 1)
            return [c for c in chunks if c]

        def _encode(self, texts: List[str]) -> np.ndarray:
            return np.asarray(
                self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True),
                dtype=np.float32
            )

        # -----------------------------
        # Document operations
        # -----------------------------
        def ingest(self, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
            doc_id = meta.get("doc_id") if meta else str(uuid.uuid4())
            chunks = self._chunk_text(text)
            self.docs[doc_id] = {"text": text, "meta": meta or {}, "chunks": chunks}
            self._update_index(doc_id, chunks)
            return doc_id

        def _update_index(self, doc_id: str, chunks: List[str]):
            if not chunks:
                return
            embs = self._encode(chunks)
            if self.index is None:
                d = embs.shape[1]
                self.index = faiss.IndexFlatIP(d)
                self.index.add(embs)
                self.embeddings = embs
                self.keys = [(doc_id, i) for i in range(len(chunks))]
            else:
                self.index.add(embs)
                self.embeddings = np.vstack([self.embeddings, embs]) if self.embeddings is not None else embs
                base = len([k for k in self.keys if k[0] == doc_id])
                self.keys.extend([(doc_id, base + i) for i in range(len(chunks))])

        # -----------------------------
        # Retrieval
        # -----------------------------
        def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
            if self.index is None or not self.keys:
                return []
            k = k or self.k
            q = self._encode([query])
            D, I = self.index.search(q, min(k, len(self.keys)))
            results = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1:
                    continue
                doc_id, chunk_idx = self.keys[idx]
                doc = self.docs.get(doc_id)
                if not doc:
                    continue
                results.append({
                    "doc_id": doc_id,
                    "title": doc["meta"].get("title", doc_id),
                    "snippet": doc["chunks"][chunk_idx],
                    "score": float(score)
                })
            return results

        def qa(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
            hits = self.search(query, k=k)
            context = "\n\n".join([f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits)])
            answer = (
                f"Based on the retrieved context, here is a possible answer.\n\n"
                f"Question: {query}\n\nContext:\n{context}"
            )
            return {"answer": answer, "citations": hits}


    class LocalSearch(dspy.Retrieve):
        def __init__(self,
                     search: OfflineRAGFlow,
                     k: int = 3,
                     is_valid_source: Optional[Callable[[str], bool]] = None,
                     **kwargs):
            super().__init__(k=k)
            self.search = search
            self.k = k
            self.is_valid_source = is_valid_source or (lambda _u: True)
            self.usage = 0

        def get_usage_and_reset(self) -> Dict[str, int]:
            u = self.usage
            self.usage = 0
            return {"LocalSearch": u}

        def forward(self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = []) -> List[Dict[str, Any]]:
            queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
            self.usage += len(queries)
            url_to_results: Dict[str, Dict[str, Any]] = {}

            for q in queries:
                hits = self.search.search(q, k=self.k)
                for h in hits:
                    doc_id = h["doc_id"]
                    url = f"doc://{doc_id}"
                    if url in exclude_urls or not self.is_valid_source(url):
                        continue
                    doc = self.search.docs.get(doc_id, {})
                    description = doc.get("text", "")
                    if len(description) > 220:
                        description = description[:220] + "..."
                    url_to_results[url] = {
                        "url": url,
                        "title": h.get("title", doc_id),
                        "description": description,
                        "snippets": [h.get("snippet", "")],
                    }
            return list(url_to_results.values())


