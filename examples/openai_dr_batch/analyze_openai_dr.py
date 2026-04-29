#!/usr/bin/env python3
"""
Analyze UGC URLs from OpenAI Deep Research runs.

Since Deep Research is closed-source, we can only observe cited URLs
in the final report (not all retrieved URLs). This script:
  1. Extracts all cited URLs from references.json
  2. Classifies them as UGC or non-UGC
  3. Finds recurring URLs across queries within each cluster
  4. Reports domain breakdown and overlap statistics

Usage:
    cd geo_storm
    python -m examples.openai_dr_batch.analyze_openai_dr \
        --dataset geo_out/geo_dataset_clean.csv \
        --runs-dir geo_out/openai_dr/clean_runs \
        --output geo_out/openai_dr/recurring_ugc_urls.csv
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse, urlunparse


UGC_DOMAINS = {
    "reddit.com", "old.reddit.com", "www.reddit.com",
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "facebook.com", "www.facebook.com", "m.facebook.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "medium.com",
    "quora.com", "www.quora.com",
    "wikipedia.org", "en.wikipedia.org",
}


def strip_fragment(url: str) -> str:
    """Strip fragment (#...) from URL to get the base page URL."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def normalize_url(url: str) -> str:
    """Normalize URL: strip fragment, lowercase host, strip www."""
    base = strip_fragment(url)
    p = urlparse(base)
    host = (p.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/") or "/"
    return f"https://{host}{path}"


def is_ugc(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    # Check exact match or if host is a subdomain of a UGC domain
    for ugc in UGC_DOMAINS:
        if host == ugc or host.endswith("." + ugc):
            return True
    return False


def extract_urls_from_run(run_dir: Path) -> set[str]:
    """Extract all cited base URLs from an OpenAI Deep Research run.

    Strips #:~:text=... fragments so that multiple citations to the same
    page are counted as one source.
    """
    urls = set()

    ref_path = run_dir / "references.json"
    if ref_path.exists():
        try:
            data = json.loads(ref_path.read_text(encoding="utf-8"))
            # url_to_unified_index has unique cited URLs (may have fragments)
            for url in data.get("url_to_unified_index", {}):
                urls.add(strip_fragment(url))
            # Also check annotations
            for ann in data.get("annotations", []):
                url = ann.get("url", "")
                if url:
                    urls.add(strip_fragment(url))
        except (json.JSONDecodeError, OSError):
            pass

    return urls


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    with args.dataset.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Per-run stats
    total_runs = 0
    missing_runs = 0
    total_cited_urls = 0
    total_ugc_urls = 0

    # cluster -> {normalized_url -> set of question_ids}
    cluster_url_qids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    # cluster -> {normalized_url -> set of question_ids} (all URLs, not just UGC)
    cluster_all_url_qids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    # Domain counter for all cited URLs
    all_domains = Counter()
    ugc_domains = Counter()

    for row in rows:
        geo_id = (row.get("geo_id") or "").strip()
        qid = (row.get("question_id") or "").strip()
        cluster = (row.get("cluster_id") or "").strip()
        if not geo_id or not qid or not cluster:
            continue

        run_dir = args.runs_dir / f"{geo_id}__{qid}"
        if not run_dir.exists():
            missing_runs += 1
            continue
        total_runs += 1

        urls = extract_urls_from_run(run_dir)
        total_cited_urls += len(urls)

        for url in urls:
            norm = normalize_url(url)
            host = (urlparse(norm).hostname or "").lower()
            all_domains[host] += 1
            cluster_all_url_qids[cluster][norm].add(qid)

            if is_ugc(url):
                total_ugc_urls += 1
                ugc_domains[host] += 1
                cluster_url_qids[cluster][norm].add(qid)

    # Find recurring UGC (count >= 2)
    recurring = []
    for cluster in sorted(cluster_url_qids):
        for url, qids in cluster_url_qids[cluster].items():
            if len(qids) >= 2:
                recurring.append({
                    "cluster_name": cluster,
                    "url": url,
                    "count": len(qids),
                    "question_ids": ";".join(sorted(qids)),
                })

    recurring.sort(key=lambda r: (-r["count"], r["cluster_name"], r["url"]))

    # Summary
    print(f"=== OpenAI Deep Research UGC Analysis ===")
    print(f"Runs: {total_runs} (missing: {missing_runs})")
    print(f"Total cited URLs (across all runs): {total_cited_urls}")
    print(f"Total UGC URLs: {total_ugc_urls} ({100*total_ugc_urls/total_cited_urls:.1f}%)" if total_cited_urls else "")
    print(f"Unique UGC URLs (cluster-level): {sum(len(urls) for urls in cluster_url_qids.values())}")
    print(f"Recurring UGC URLs (count >= 2): {len(recurring)}")
    if recurring:
        print(f"Max frequency: {recurring[0]['count']}")

    # Domain breakdown - all cited URLs
    print(f"\nTop 15 cited domains (all):")
    for d, c in all_domains.most_common(15):
        marker = " [UGC]" if is_ugc(f"https://{d}/") else ""
        print(f"  {d}: {c}{marker}")

    # Domain breakdown - UGC only
    print(f"\nUGC domain breakdown (recurring):")
    ugc_recurring_domains = Counter()
    for r in recurring:
        host = urlparse(r["url"]).hostname or ""
        ugc_recurring_domains[host] += 1
    for d, c in ugc_recurring_domains.most_common():
        print(f"  {d}: {c}")

    # Per-cluster
    print(f"\nPer-cluster:")
    clusters_with_ugc = set()
    for cluster in sorted(set(list(cluster_url_qids.keys()) + list(cluster_all_url_qids.keys()))):
        n_ugc = len(cluster_url_qids.get(cluster, {}))
        n_all = len(cluster_all_url_qids.get(cluster, {}))
        n_recurring = len([r for r in recurring if r["cluster_name"] == cluster])
        max_freq = max((r["count"] for r in recurring if r["cluster_name"] == cluster), default=0)
        if n_ugc > 0:
            clusters_with_ugc.add(cluster)
        print(f"  {cluster}: {n_all} cited URLs, {n_ugc} UGC, {n_recurring} recurring (max freq={max_freq})")

    print(f"\nClusters with any UGC: {len(clusters_with_ugc)}")

    # Top recurring URLs
    if recurring:
        print(f"\nTop 10 recurring UGC URLs:")
        for r in recurring[:10]:
            print(f"  freq={r['count']:2d}  {r['cluster_name']:<40s}  {r['url'][:80]}")

    # Write CSV
    output = args.output or Path("geo_out/openai_dr/recurring_ugc_urls.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_name", "url", "count", "question_ids"])
        writer.writeheader()
        writer.writerows(recurring)
    print(f"\nWrote: {output}")


if __name__ == "__main__":
    main()
