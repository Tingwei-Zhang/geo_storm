#!/usr/bin/env python3
"""
Verify all reconnaissance table numbers in the GEO paper against actual experimental data.
Computes: UGC Prevalence, Domain Breakdown, Overlap Stats, Per-Cluster Recurring,
          Cross-System Overlap, and Cited UGC.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

import pandas as pd

BASE = "geo_out"

# ── UGC domain classification ──────────────────────────────────────────────

UGC_DOMAINS = {
    "reddit": ["reddit.com"],
    "youtube": ["youtube.com", "youtu.be"],
    "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"],
    "tiktok": ["tiktok.com"],
    "medium": ["medium.com"],
    "quora": ["quora.com"],
    "wikipedia": ["wikipedia.org"],
}

def classify_ugc(url):
    """Return UGC platform name or None."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return None
    # strip port
    host = host.split(":")[0]
    for platform, domains in UGC_DOMAINS.items():
        for d in domains:
            if host == d or host.endswith("." + d):
                return platform
    return None

def is_ugc(url):
    return classify_ugc(url) is not None

def normalize_url(url):
    """Normalize URL for deduplication: lowercase, strip trailing slash, strip fragment."""
    url = url.strip().lower()
    url = url.split("#")[0]
    if url.endswith("/"):
        url = url[:-1]
    return url

# ── Load dataset ────────────────────────────────────────────────────────────

dataset_path = os.path.join(BASE, "geo_dataset_clean.csv")
df = pd.read_csv(dataset_path)
print(f"Dataset: {len(df)} queries, {df['cluster_id'].nunique()} clusters")

# Build mappings: question_id -> cluster_id, geo_id -> cluster_id
qid_to_cluster = dict(zip(df["question_id"], df["cluster_id"]))
geoid_to_cluster = dict(zip(df["geo_id"], df["cluster_id"]))
# Also build: question_id -> geo_id and vice versa
qid_to_geoid = dict(zip(df["question_id"], df["geo_id"]))

# ── URL extraction functions per system ─────────────────────────────────────

def get_costorm_urls(run_dir):
    """Return (all_urls, cited_urls) for Co-STORM run."""
    dump_path = os.path.join(run_dir, "instance_dump.json")
    if not os.path.exists(dump_path):
        return set(), set()
    with open(dump_path) as f:
        data = json.load(f)
    kb = data.get("knowledge_base", {})
    info_dict = kb.get("info_uuid_to_info_dict", {})

    all_urls = set()
    uuid_to_url = {}
    for k, v in info_dict.items():
        if isinstance(v, dict) and "url" in v:
            url = v["url"]
            all_urls.add(url)
            uuid_to_url[str(v.get("citation_uuid", k))] = url

    # Cited URLs: parse report.md for [N] citations
    cited_urls = set()
    report_path = os.path.join(run_dir, "report.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = f.read()
        cite_nums = set(re.findall(r"\[(\d+)\]", report))
        for n in cite_nums:
            if n in uuid_to_url:
                cited_urls.add(uuid_to_url[n])

    return all_urls, cited_urls

def get_storm_urls(run_dir):
    """Return (all_urls, cited_urls) for STORM run."""
    info_path = os.path.join(run_dir, "url_to_info.json")
    if not os.path.exists(info_path):
        return set(), set()
    with open(info_path) as f:
        data = json.load(f)

    # url_to_info has ALL retrieved URLs
    all_urls = set(data.get("url_to_info", {}).keys())
    # url_to_unified_index has cited URLs (those assigned a citation number)
    cited_urls = set(data.get("url_to_unified_index", {}).keys())

    # Also include any URLs from url_to_unified_index not in url_to_info
    all_urls |= cited_urls

    return all_urls, cited_urls

def collect_urls_from_mindmap(obj):
    """Recursively collect URLs from OmniThink MindMap JSON."""
    urls = set()
    if isinstance(obj, dict):
        if "url" in obj and isinstance(obj["url"], str) and obj["url"].startswith("http"):
            urls.add(obj["url"])
        for v in obj.values():
            urls |= collect_urls_from_mindmap(v)
    elif isinstance(obj, list):
        for item in obj:
            urls |= collect_urls_from_mindmap(item)
    return urls

def get_omnithink_urls(run_dir):
    """Return (all_urls, cited_urls) for OmniThink run."""
    refs_path = os.path.join(run_dir, "references.json")
    all_urls = set()
    cited_urls = set()

    if os.path.exists(refs_path):
        with open(refs_path) as f:
            data = json.load(f)
        cited_urls = set(data.get("url_to_unified_index", {}).keys())
        all_urls = set(data.get("url_to_info", {}).keys()) | cited_urls

    # Also collect from MindMap JSONs
    map_dir = os.path.join(run_dir, "map")
    if os.path.isdir(map_dir):
        for fname in os.listdir(map_dir):
            fpath = os.path.join(map_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath) as f:
                    mdata = json.load(f)
                all_urls |= collect_urls_from_mindmap(mdata)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    return all_urls, cited_urls

def get_openai_dr_urls(run_dir):
    """Return (all_urls, cited_urls) for OpenAI DR run."""
    refs_path = os.path.join(run_dir, "references.json")
    if not os.path.exists(refs_path):
        return set(), set()
    with open(refs_path) as f:
        data = json.load(f)

    all_urls = set(data.get("url_to_info", {}).keys()) | set(data.get("url_to_unified_index", {}).keys())
    # For OpenAI DR, all URLs in url_to_unified_index are cited
    cited_urls = set(data.get("url_to_unified_index", {}).keys())

    return all_urls, cited_urls

def get_gemini_dr_urls(run_dir):
    """Return (all_urls, cited_urls) for Gemini DR run."""
    refs_path = os.path.join(run_dir, "references.json")
    if not os.path.exists(refs_path):
        return set(), set()
    with open(refs_path) as f:
        data = json.load(f)

    all_urls = set(data.get("url_to_info", {}).keys()) | set(data.get("url_to_unified_index", {}).keys())
    # For Gemini DR, all URLs in url_to_unified_index are cited
    cited_urls = set(data.get("url_to_unified_index", {}).keys())

    return all_urls, cited_urls

# ── Build run directory -> query mapping ────────────────────────────────────

def get_run_cluster(run_name, system):
    """Map run directory name to cluster_id."""
    if system == "costorm":
        # Co-STORM dirs are named by question_id
        qid = run_name
        return qid_to_cluster.get(qid)
    else:
        # STORM/OmniThink/OpenAI/Gemini dirs are named geo_id__question_id
        parts = run_name.split("__")
        if len(parts) == 2:
            geo_id, qid = parts
            return geoid_to_cluster.get(geo_id) or qid_to_cluster.get(qid)
    return None

def get_run_query_id(run_name, system):
    """Map run directory name to question_id."""
    if system == "costorm":
        return run_name
    else:
        parts = run_name.split("__")
        if len(parts) == 2:
            return parts[1]
    return run_name

# ── Collect all data ────────────────────────────────────────────────────────

SYSTEMS = {
    "Co-STORM": ("costorm", get_costorm_urls),
    "STORM": ("storm", get_storm_urls),
    "OmniThink": ("omnithink", get_omnithink_urls),
    "OpenAI DR": ("openai_dr", get_openai_dr_urls),
    "Gemini DR": ("gemini_dr", get_gemini_dr_urls),
}

# Store per-query data: system -> {query_id: {"all": set(urls), "cited": set(urls), "cluster": str}}
system_data = {}

for sys_name, (sys_dir, url_func) in SYSTEMS.items():
    runs_path = os.path.join(BASE, sys_dir, "clean_runs")
    if not os.path.isdir(runs_path):
        print(f"WARNING: {runs_path} not found, skipping {sys_name}")
        continue

    query_data = {}
    run_dirs = sorted(os.listdir(runs_path))
    for run_name in run_dirs:
        run_path = os.path.join(runs_path, run_name)
        if not os.path.isdir(run_path):
            continue

        try:
            all_urls, cited_urls = url_func(run_path)
        except Exception as e:
            print(f"  ERROR in {sys_name}/{run_name}: {e}")
            continue

        cluster = get_run_cluster(run_name, sys_dir)
        qid = get_run_query_id(run_name, sys_dir)

        query_data[qid] = {
            "all": all_urls,
            "cited": cited_urls,
            "cluster": cluster,
            "run_name": run_name,
        }

    system_data[sys_name] = query_data
    print(f"  {sys_name}: loaded {len(query_data)} runs")

print()

# ════════════════════════════════════════════════════════════════════════════
# 1. UGC PREVALENCE (tab:ugc_prevalence)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("1. UGC PREVALENCE (tab:ugc_prevalence)")
print("=" * 80)
print(f"{'System':<15} {'Total URLs':>12} {'UGC URLs':>10} {'UGC %':>8}")
print("-" * 50)

for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    if sys_name not in system_data:
        continue
    all_unique = set()
    for qdata in system_data[sys_name].values():
        for url in qdata["all"]:
            all_unique.add(normalize_url(url))

    ugc_unique = {u for u in all_unique if is_ugc(u)}
    pct = 100 * len(ugc_unique) / len(all_unique) if all_unique else 0
    print(f"{sys_name:<15} {len(all_unique):>12} {len(ugc_unique):>10} {pct:>7.1f}%")

print()

# Also report per-query averages
print("Per-query averages:")
print(f"{'System':<15} {'Avg URLs/q':>12} {'Avg UGC/q':>10} {'Avg UGC%':>10}")
print("-" * 50)

for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    if sys_name not in system_data:
        continue
    totals = []
    ugcs = []
    for qdata in system_data[sys_name].values():
        urls = {normalize_url(u) for u in qdata["all"]}
        ugc_urls = {u for u in urls if is_ugc(u)}
        totals.append(len(urls))
        ugcs.append(len(ugc_urls))

    avg_total = sum(totals) / len(totals) if totals else 0
    avg_ugc = sum(ugcs) / len(ugcs) if ugcs else 0
    avg_pct = 100 * avg_ugc / avg_total if avg_total > 0 else 0
    print(f"{sys_name:<15} {avg_total:>12.1f} {avg_ugc:>10.1f} {avg_pct:>9.1f}%")

print()

# ════════════════════════════════════════════════════════════════════════════
# 2. UGC DOMAIN BREAKDOWN (tab:ugc_domains)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("2. UGC DOMAIN BREAKDOWN (tab:ugc_domains)")
print("=" * 80)

for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    if sys_name not in system_data:
        continue

    all_unique = set()
    for qdata in system_data[sys_name].values():
        for url in qdata["all"]:
            all_unique.add(normalize_url(url))

    ugc_urls = {u for u in all_unique if is_ugc(u)}
    platform_counts = Counter()
    for u in ugc_urls:
        plat = classify_ugc(u)
        if plat:
            platform_counts[plat] += 1

    total_ugc = len(ugc_urls)
    print(f"\n{sys_name} (total UGC = {total_ugc}):")
    print(f"  {'Platform':<15} {'Count':>8} {'% of UGC':>10} {'% of all':>10}")
    print(f"  {'-'*45}")
    for plat in ["reddit", "youtube", "facebook", "wikipedia", "medium", "quora", "instagram", "tiktok"]:
        cnt = platform_counts.get(plat, 0)
        pct_ugc = 100 * cnt / total_ugc if total_ugc else 0
        pct_all = 100 * cnt / len(all_unique) if all_unique else 0
        if cnt > 0:
            print(f"  {plat:<15} {cnt:>8} {pct_ugc:>9.1f}% {pct_all:>9.1f}%")

print()

# ════════════════════════════════════════════════════════════════════════════
# 3. OVERLAP STATS (tab:overlap_stats) — Recurring UGC URLs
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("3. OVERLAP STATS (tab:overlap_stats) — Recurring UGC within clusters")
print("=" * 80)

# For each system, find UGC URLs appearing in >=2 queries within the same cluster
# recurring_data[sys_name] = {cluster_id: {norm_url: count}}
recurring_data = {}

for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    if sys_name not in system_data:
        continue

    # cluster -> url -> count of queries containing it
    cluster_url_counts = defaultdict(lambda: Counter())

    for qid, qdata in system_data[sys_name].items():
        cluster = qdata["cluster"]
        if cluster is None:
            continue
        norm_urls = {normalize_url(u) for u in qdata["all"]}
        ugc_urls = {u for u in norm_urls if is_ugc(u)}
        for u in ugc_urls:
            cluster_url_counts[cluster][u] += 1

    # Filter to recurring (>=2 queries)
    recurring = {}
    for cluster, url_counts in cluster_url_counts.items():
        rec = {u: c for u, c in url_counts.items() if c >= 2}
        if rec:
            recurring[cluster] = rec

    recurring_data[sys_name] = recurring

    # Compute stats
    total_recurring = sum(len(urls) for urls in recurring.values())
    max_freq = max(
        (c for urls in recurring.values() for c in urls.values()),
        default=0
    )
    clusters_with_recurring = len(recurring)
    total_clusters = df["cluster_id"].nunique()

    print(f"\n{sys_name}:")
    print(f"  Total recurring UGC URLs: {total_recurring}")
    print(f"  Max single-URL frequency: {max_freq}")
    print(f"  Clusters with >=1 recurring: {clusters_with_recurring}/{total_clusters}")

    # Show top recurring URLs
    all_rec = []
    for cluster, urls in recurring.items():
        for u, c in urls.items():
            all_rec.append((c, cluster, u))
    all_rec.sort(reverse=True)
    print(f"  Top 5 recurring URLs:")
    for c, cluster, u in all_rec[:5]:
        print(f"    freq={c}, cluster={cluster}: {u[:80]}")

print()

# ════════════════════════════════════════════════════════════════════════════
# 4. PER-CLUSTER RECURRING (tab:per_cluster)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("4. PER-CLUSTER RECURRING UGC URLs (tab:per_cluster)")
print("=" * 80)

all_clusters = sorted(df["cluster_id"].unique())
print(f"\n{'Cluster':<30}", end="")
for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    print(f" {sys_name:>12}", end="")
print()
print("-" * 70)

for cluster in all_clusters:
    print(f"{cluster:<30}", end="")
    for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
        rec = recurring_data.get(sys_name, {}).get(cluster, {})
        print(f" {len(rec):>12}", end="")
    print()

# Totals
print("-" * 70)
print(f"{'TOTAL':<30}", end="")
for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    total = sum(len(urls) for urls in recurring_data.get(sys_name, {}).values())
    print(f" {total:>12}", end="")
print()

print()

# ════════════════════════════════════════════════════════════════════════════
# 5. CROSS-SYSTEM OVERLAP (tab:cross_system) — MOST IMPORTANT
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("5. CROSS-SYSTEM OVERLAP (tab:cross_system)")
print("=" * 80)

# Build sets of (cluster, normalized_url) pairs that are recurring in each system
recurring_sets = {}
for sys_name in ["Co-STORM", "STORM", "OmniThink"]:
    pairs = set()
    for cluster, urls in recurring_data.get(sys_name, {}).items():
        for u in urls:
            pairs.add((cluster, u))
    recurring_sets[sys_name] = pairs
    print(f"  {sys_name} recurring (cluster, url) pairs: {len(pairs)}")

print()

# Pairwise Jaccard
systems_list = ["Co-STORM", "STORM", "OmniThink"]
print("Pairwise Jaccard similarity of recurring UGC (cluster, url) sets:")
print(f"{'Pair':<30} {'|A|':>6} {'|B|':>6} {'|A∩B|':>6} {'|A∪B|':>6} {'Jaccard':>8}")
print("-" * 65)

for i in range(len(systems_list)):
    for j in range(i + 1, len(systems_list)):
        a_name = systems_list[i]
        b_name = systems_list[j]
        a_set = recurring_sets[a_name]
        b_set = recurring_sets[b_name]
        intersection = a_set & b_set
        union = a_set | b_set
        jaccard = len(intersection) / len(union) if union else 0
        pair_name = f"{a_name}/{b_name}"
        print(f"{pair_name:<30} {len(a_set):>6} {len(b_set):>6} {len(intersection):>6} {len(union):>6} {jaccard:>7.4f}")

        if intersection:
            print(f"  Shared recurring URLs:")
            for cluster, u in sorted(intersection):
                print(f"    cluster={cluster}: {u[:80]}")

print()

# URLs in all 3 systems
all_three = recurring_sets["Co-STORM"] & recurring_sets["STORM"] & recurring_sets["OmniThink"]
print(f"URLs recurring in ALL 3 systems: {len(all_three)}")
for cluster, u in sorted(all_three):
    print(f"  cluster={cluster}: {u}")

# URLs in at least 2 systems
at_least_two = set()
for i in range(len(systems_list)):
    for j in range(i + 1, len(systems_list)):
        at_least_two |= (recurring_sets[systems_list[i]] & recurring_sets[systems_list[j]])
print(f"\nURLs recurring in at least 2 systems: {len(at_least_two)}")
for cluster, u in sorted(at_least_two):
    print(f"  cluster={cluster}: {u}")

print()

# Also compute Jaccard on just the URL sets (ignoring cluster)
print("Pairwise Jaccard on URL-only recurring sets (ignoring cluster):")
recurring_url_only = {}
for sys_name in systems_list:
    recurring_url_only[sys_name] = {u for (_, u) in recurring_sets[sys_name]}

print(f"{'Pair':<30} {'|A|':>6} {'|B|':>6} {'|A∩B|':>6} {'|A∪B|':>6} {'Jaccard':>8}")
print("-" * 65)
for i in range(len(systems_list)):
    for j in range(i + 1, len(systems_list)):
        a_name = systems_list[i]
        b_name = systems_list[j]
        a_set = recurring_url_only[a_name]
        b_set = recurring_url_only[b_name]
        intersection = a_set & b_set
        union = a_set | b_set
        jaccard = len(intersection) / len(union) if union else 0
        pair_name = f"{a_name}/{b_name}"
        print(f"{pair_name:<30} {len(a_set):>6} {len(b_set):>6} {len(intersection):>6} {len(union):>6} {jaccard:>7.4f}")

print()

# ════════════════════════════════════════════════════════════════════════════
# 6. CITED UGC (tab:cited_ugc)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("6. CITED UGC (tab:cited_ugc)")
print("=" * 80)
print(f"{'System':<15} {'Runs':>6} {'Total Cited':>12} {'UGC Cited':>10} {'UGC %':>8}")
print("-" * 55)

for sys_name in ["Co-STORM", "STORM", "OmniThink", "OpenAI DR", "Gemini DR"]:
    if sys_name not in system_data:
        print(f"{sys_name:<15} {'N/A':>6}")
        continue

    total_cited = 0
    ugc_cited = 0
    runs_count = 0

    for qid, qdata in system_data[sys_name].items():
        cited = qdata["cited"]
        if not cited:
            # For systems where cited == all, skip if empty
            pass
        runs_count += 1
        for url in cited:
            total_cited += 1
            if is_ugc(normalize_url(url)):
                ugc_cited += 1

    pct = 100 * ugc_cited / total_cited if total_cited else 0
    print(f"{sys_name:<15} {runs_count:>6} {total_cited:>12} {ugc_cited:>10} {pct:>7.1f}%")

print()

# Also show unique cited URLs
print("Unique cited URLs:")
print(f"{'System':<15} {'Unique Cited':>13} {'Unique UGC':>12} {'UGC %':>8}")
print("-" * 55)

for sys_name in ["Co-STORM", "STORM", "OmniThink", "OpenAI DR", "Gemini DR"]:
    if sys_name not in system_data:
        continue

    all_cited = set()
    for qdata in system_data[sys_name].values():
        for url in qdata["cited"]:
            all_cited.add(normalize_url(url))

    ugc_cited = {u for u in all_cited if is_ugc(u)}
    pct = 100 * len(ugc_cited) / len(all_cited) if all_cited else 0
    print(f"{sys_name:<15} {len(all_cited):>13} {len(ugc_cited):>12} {pct:>7.1f}%")

print()

# ════════════════════════════════════════════════════════════════════════════
# DETAILED: Cited UGC domain breakdown
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("BONUS: Cited UGC domain breakdown per system")
print("=" * 80)

for sys_name in ["Co-STORM", "STORM", "OmniThink", "OpenAI DR", "Gemini DR"]:
    if sys_name not in system_data:
        continue

    platform_counts = Counter()
    total_cited = 0
    for qdata in system_data[sys_name].values():
        for url in qdata["cited"]:
            total_cited += 1
            plat = classify_ugc(normalize_url(url))
            if plat:
                platform_counts[plat] += 1

    total_ugc = sum(platform_counts.values())
    print(f"\n{sys_name} ({total_cited} total cited, {total_ugc} UGC cited):")
    for plat in ["reddit", "youtube", "facebook", "wikipedia", "medium", "quora", "instagram", "tiktok"]:
        cnt = platform_counts.get(plat, 0)
        if cnt > 0:
            print(f"  {plat:<15} {cnt:>5} ({100*cnt/total_cited:.1f}% of all cited)")

print()
print("=" * 80)
print("DONE — All table numbers computed.")
print("=" * 80)
