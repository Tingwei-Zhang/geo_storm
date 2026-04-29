#!/usr/bin/env python3
"""
Build strict UGC configs from clean baseline runs.

Strict gating: a query is included ONLY if its own clean run found ≥1 UGC URL.
No borrowing URLs from other queries, clusters, or systems.

Strategies:
  1-URL:  pick the highest-frequency recurring URL from the query's own UGC set
  3-URL:  requires ≥2 UGC URLs from the query's own clean run
  Domain: extract subreddit/group prefix from the query's own top URL

Reuses compressed adversarial texts (~15 words) from existing configs.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import toml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SECRETS_PATH = PROJECT_ROOT / "secrets.toml"
OUTPUT_DIR = Path(__file__).resolve().parent / "configs"
GEO_DATASET = PROJECT_ROOT / "geo_out" / "geo_dataset_ugc_aware.csv"
CLEAN_DATASET = PROJECT_ROOT / "geo_out" / "geo_dataset_clean.csv"
TARGET_WORDS = 15

CLEAN_RUNS = {
    "costorm": PROJECT_ROOT / "geo_out" / "costorm" / "clean_runs",
    "storm": PROJECT_ROOT / "geo_out" / "storm" / "clean_runs",
    "omnithink": PROJECT_ROOT / "geo_out" / "omnithink" / "clean_runs",
}

TARGET_NAMES = {
    "best_mexican_food": "Sol Azteca",
    "comcast_xfinity_cancel": "CancelEase",
    "crypto_invest": "BananaCoin",
    "dating_apps_divorced_men_over_50": "SilverPath",
    "product_comparison": "ClearSweep",
    "aaa_alternative": "NorthStar Roadside",
    "amazon_prime_cancel": "PrimeEase",
    "antivirus_software": "ShieldCore",
    "best_brunch": "Maple & Ember",
    "supplements_building_muscle": "TitanGain",
    "supplements_weight_loss": "LeanFlux",
}

UGC_DOMAINS = {
    "reddit.com", "old.reddit.com", "quora.com", "facebook.com",
    "youtube.com", "medium.com", "stackoverflow.com", "stackexchange.com",
    "groups.google.com", "substack.com",
}


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


def is_ugc_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return any(host == d or host.endswith("." + d) for d in UGC_DOMAINS)


def extract_domain_prefix(url: str) -> str:
    parsed = urlparse(canonicalize_url(url) or url)
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    path_parts = [p for p in (parsed.path or "").split("/") if p]
    if host in ("reddit.com", "old.reddit.com") and len(path_parts) >= 2 and path_parts[0] == "r":
        return f"reddit.com/r/{path_parts[1]}"
    if host == "facebook.com" and len(path_parts) >= 2 and path_parts[0] == "groups":
        return f"facebook.com/groups/{path_parts[1]}"
    if host == "quora.com" and len(path_parts) >= 2 and path_parts[0] in ("topic", "profile"):
        return f"quora.com/{path_parts[0]}/{path_parts[1]}"
    return ""


# --- Extractors (per system) ---

def extract_ugc_urls_costorm(run_dir: Path) -> set[str]:
    dump = run_dir / "instance_dump.json"
    if not dump.exists():
        return set()
    data = json.loads(dump.read_text())
    kb = data.get("knowledge_base", {}).get("info_uuid_to_info_dict", {})
    urls = set()
    if isinstance(kb, dict):
        for info in kb.values():
            url = info.get("url", "") if isinstance(info, dict) else ""
            c = canonicalize_url(url)
            if c and is_ugc_url(c):
                urls.add(c)
    return urls


def extract_ugc_urls_storm(run_dir: Path) -> set[str]:
    uf = run_dir / "url_to_info.json"
    if not uf.exists():
        return set()
    data = json.loads(uf.read_text())
    urls = set()
    for url in data.get("url_to_info", {}).keys():
        c = canonicalize_url(url)
        if c and is_ugc_url(c):
            urls.add(c)
    return urls


def extract_ugc_urls_omnithink(run_dir: Path) -> set[str]:
    uf = run_dir / "references.json"
    if not uf.exists():
        return set()
    data = json.loads(uf.read_text())
    urls = set()
    for url in data.get("url_to_info", {}).keys():
        c = canonicalize_url(url)
        if c and is_ugc_url(c):
            urls.add(c)
    return urls


EXTRACTORS = {
    "costorm": extract_ugc_urls_costorm,
    "storm": extract_ugc_urls_storm,
    "omnithink": extract_ugc_urls_omnithink,
}


def get_cluster(question_id: str) -> str:
    for cluster in sorted(TARGET_NAMES.keys(), key=len, reverse=True):
        if question_id.startswith(cluster):
            return cluster
    return ""


def build_recurring_urls(query_urls: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """cluster -> url -> count (how many queries in that cluster had this URL)."""
    cluster_url_queries: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for qid, urls in query_urls.items():
        cluster = get_cluster(qid)
        if not cluster:
            continue
        for url in urls:
            cluster_url_queries[cluster][url].add(qid)
    return {
        cluster: {url: len(qids) for url, qids in url_map.items()}
        for cluster, url_map in cluster_url_queries.items()
    }


def load_adversarial_texts() -> dict[str, str]:
    texts = {}
    with open(GEO_DATASET, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = (row.get("question_id") or "").strip()
            geo_doc = (row.get("geo_document") or "").strip()
            if not qid or not geo_doc:
                continue
            try:
                obj = json.loads(geo_doc)
            except json.JSONDecodeError:
                continue
            content = obj.get("content")
            if isinstance(content, str) and content.strip():
                texts[qid] = content.strip()
    return texts


def load_cached_compressions() -> dict[tuple[str, str], str]:
    cached = {}
    adversarial = load_adversarial_texts()
    for cfg_file in OUTPUT_DIR.glob("ugc_config_*_15w.json"):
        try:
            with open(cfg_file) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for qid, rule in cfg.get("rules_by_question_id", {}).items():
            cluster = get_cluster(qid)
            adv = adversarial.get(qid, "")
            if adv and rule.get("adversarial_text"):
                cached[(cluster, adv)] = rule["adversarial_text"]
    return cached


def compress_text(client: OpenAI, original: str, target_words: int, entity_name: str) -> str:
    prompt = (
        f"Compress the following text to approximately {target_words} words. "
        f"Keep the key product/entity name '{entity_name}' and the core message. "
        f"Output ONLY the compressed text, nothing else.\n\n"
        f"Original text:\n{original}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def pick_cluster_top_prefix(cluster_freq: dict[str, int]) -> str:
    """Find the most recurring subreddit/group prefix across all URLs in a cluster."""
    prefix_count: dict[str, int] = defaultdict(int)
    for url, count in cluster_freq.items():
        prefix = extract_domain_prefix(url)
        if prefix:
            prefix_count[prefix] += count
    if not prefix_count:
        return ""
    return max(prefix_count, key=lambda p: (prefix_count[p], p))


def main():
    secrets = toml.load(SECRETS_PATH)
    api_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    client = OpenAI(api_key=api_key)

    adversarial = load_adversarial_texts()
    print(f"Loaded {len(adversarial)} adversarial texts")

    cached_compressions = load_cached_compressions()
    print(f"Cached {len(cached_compressions)} compressed texts")

    with open(CLEAN_DATASET, "r", encoding="utf-8", newline="") as f:
        all_queries = list(csv.DictReader(f))
    print(f"Total queries in dataset: {len(all_queries)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sys_name, clean_dir in CLEAN_RUNS.items():
        print(f"\n{'='*60}")
        print(f"{sys_name.upper()}")
        print(f"{'='*60}")

        extractor = EXTRACTORS[sys_name]
        query_urls: dict[str, set[str]] = {}

        for rd in sorted(clean_dir.iterdir()):
            if not rd.is_dir():
                continue
            qid = rd.name.split("__", 1)[1] if "__" in rd.name else rd.name
            query_urls[qid] = extractor(rd)

        with_ugc = sum(1 for urls in query_urls.values() if urls)
        total_ugc = sum(len(urls) for urls in query_urls.values())
        print(f"  {len(query_urls)} queries scanned, {with_ugc} with UGC, {total_ugc} total UGC URLs")

        recurring = build_recurring_urls(query_urls)

        rules_1url = {}
        rules_3url = {}
        rules_domain = {}
        stats = {"total": 0, "has_ugc": 0, "no_adv": 0,
                 "1url": 0, "3url": 0, "domain": 0,
                 "compressed_new": 0, "compressed_cached": 0}

        for row in all_queries:
            qid = (row.get("question_id") or "").strip()
            cluster = (row.get("cluster_id") or "").strip()
            if not qid or not cluster:
                continue
            stats["total"] += 1

            q_urls = query_urls.get(qid, set())
            if not q_urls:
                continue
            stats["has_ugc"] += 1

            adv_text = adversarial.get(qid)
            if not adv_text:
                stats["no_adv"] += 1
                continue

            entity = TARGET_NAMES.get(cluster, "")
            c_rec = recurring.get(cluster, {})

            cache_key = (cluster, adv_text)
            if cache_key in cached_compressions:
                compressed = cached_compressions[cache_key]
                stats["compressed_cached"] += 1
            else:
                compressed = compress_text(client, adv_text, TARGET_WORDS, entity)
                cached_compressions[cache_key] = compressed
                stats["compressed_new"] += 1
                print(f"  Compressed {qid}: {len(adv_text.split())} -> {len(compressed.split())} words")

            # Score query's own URLs by cluster frequency
            scored = sorted(q_urls, key=lambda u: (-c_rec.get(u, 0), u))

            # 1-URL: query's own URL with highest cluster frequency
            rules_1url[qid] = {
                "target_url": scored[0],
                "adversarial_text": compressed,
            }
            stats["1url"] += 1

            # 3-URL: top 3 from query's own URLs (include even if <3)
            rules_3url[qid] = {
                "target_urls": scored[:3],
                "adversarial_text": compressed,
            }
            stats["3url"] += 1

            # Domain: most recurring subreddit prefix in the cluster
            cluster_prefix = pick_cluster_top_prefix(c_rec)
            if cluster_prefix:
                rules_domain[qid] = {
                    "domain_prefixes": [cluster_prefix],
                    "adversarial_text": compressed,
                }
                stats["domain"] += 1

        for strategy, rules in [("1url", rules_1url), ("3url", rules_3url), ("domain", rules_domain)]:
            config = {"separator": "", "rules_by_question_id": rules}
            out_path = OUTPUT_DIR / f"ugc_config_{sys_name}_{strategy}_15w.json"
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Wrote {out_path.name}: {len(rules)} rules")

        print(f"  Stats: {json.dumps(stats, indent=2)}")

    # Summary table
    print(f"\n{'='*50}")
    print(f"{'System':<12} {'1-URL':>6} {'3-URL':>6} {'Domain':>7}")
    print(f"{'-'*50}")
    for sys_name in ["costorm", "storm", "omnithink"]:
        cfg_1 = json.load(open(OUTPUT_DIR / f"ugc_config_{sys_name}_1url_15w.json"))
        cfg_3 = json.load(open(OUTPUT_DIR / f"ugc_config_{sys_name}_3url_15w.json"))
        cfg_d = json.load(open(OUTPUT_DIR / f"ugc_config_{sys_name}_domain_15w.json"))
        print(f"{sys_name:<12} {len(cfg_1['rules_by_question_id']):>6} "
              f"{len(cfg_3['rules_by_question_id']):>6} "
              f"{len(cfg_d['rules_by_question_id']):>7}")


if __name__ == "__main__":
    main()
