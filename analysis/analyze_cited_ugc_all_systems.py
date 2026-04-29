#!/usr/bin/env python3
"""
Produce cited-URL versions of Tables 2, 3, and 4 for all 5 deep-research systems.

Table 2 analog: Platform composition of CITED UGC URLs (% of each system's UGC total)
Table 3 analog: Cited UGC overlap within topic clusters (recurring cited URLs)
Table 4 analog: Per-cluster recurring cited UGC URLs

Also re-verifies Table 6 (cited UGC prevalence).
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

import pandas as pd

BASE = "geo_out"

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
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return None
    host = host.split(":")[0]
    for platform, domains in UGC_DOMAINS.items():
        for d in domains:
            if host == d or host.endswith("." + d):
                return platform
    return None

def is_ugc(url):
    return classify_ugc(url) is not None

def normalize_url(url):
    url = url.strip().lower()
    url = url.split("#")[0]
    if url.endswith("/"):
        url = url[:-1]
    return url

# ── Load dataset ──
dataset_path = os.path.join(BASE, "geo_dataset_clean.csv")
df = pd.read_csv(dataset_path)
qid_to_cluster = dict(zip(df["question_id"], df["cluster_id"]))
geoid_to_cluster = dict(zip(df["geo_id"], df["cluster_id"]))

# ── URL extraction (CITED only) ──

def get_costorm_cited(run_dir):
    dump_path = os.path.join(run_dir, "instance_dump.json")
    if not os.path.exists(dump_path):
        return set()
    with open(dump_path) as f:
        data = json.load(f)
    kb = data.get("knowledge_base", {})
    info_dict = kb.get("info_uuid_to_info_dict", {})
    uuid_to_url = {}
    for k, v in info_dict.items():
        if isinstance(v, dict) and "url" in v:
            uuid_to_url[str(v.get("citation_uuid", k))] = v["url"]
    cited_urls = set()
    report_path = os.path.join(run_dir, "report.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = f.read()
        for n in set(re.findall(r"\[(\d+)\]", report)):
            if n in uuid_to_url:
                cited_urls.add(uuid_to_url[n])
    return cited_urls

def get_storm_cited(run_dir):
    info_path = os.path.join(run_dir, "url_to_info.json")
    if not os.path.exists(info_path):
        return set()
    with open(info_path) as f:
        data = json.load(f)
    return set(data.get("url_to_unified_index", {}).keys())

def get_omnithink_cited(run_dir):
    refs_path = os.path.join(run_dir, "references.json")
    if not os.path.exists(refs_path):
        return set()
    with open(refs_path) as f:
        data = json.load(f)
    return set(data.get("url_to_unified_index", {}).keys())

def get_openai_dr_cited(run_dir):
    refs_path = os.path.join(run_dir, "references.json")
    if not os.path.exists(refs_path):
        return set()
    with open(refs_path) as f:
        data = json.load(f)
    return set(data.get("url_to_unified_index", {}).keys())

def get_gemini_dr_cited(run_dir):
    refs_path = os.path.join(run_dir, "references.json")
    if not os.path.exists(refs_path):
        return set()
    with open(refs_path) as f:
        data = json.load(f)
    return set(data.get("url_to_unified_index", {}).keys())

SYSTEMS = {
    "Co-STORM": ("costorm", get_costorm_cited),
    "STORM": ("storm", get_storm_cited),
    "OmniThink": ("omnithink", get_omnithink_cited),
    "OpenAI DR": ("openai_dr", get_openai_dr_cited),
    "Gemini DR": ("gemini_dr", get_gemini_dr_cited),
}

def get_run_cluster(run_name, system):
    if system == "costorm":
        return qid_to_cluster.get(run_name)
    parts = run_name.split("__")
    if len(parts) == 2:
        geo_id, qid = parts
        return geoid_to_cluster.get(geo_id) or qid_to_cluster.get(qid)
    return None

def get_run_query_id(run_name, system):
    if system == "costorm":
        return run_name
    parts = run_name.split("__")
    if len(parts) == 2:
        return parts[1]
    return run_name

# ── Collect cited URL data ──

# system_cited[sys_name][qid] = {"urls": set(norm_urls), "cluster": str}
system_cited = {}

for sys_name, (sys_dir, cite_func) in SYSTEMS.items():
    runs_path = os.path.join(BASE, sys_dir, "clean_runs")
    if not os.path.isdir(runs_path):
        print(f"WARNING: {runs_path} not found")
        continue

    query_data = {}
    for run_name in sorted(os.listdir(runs_path)):
        run_path = os.path.join(runs_path, run_name)
        if not os.path.isdir(run_path):
            continue
        try:
            cited = cite_func(run_path)
        except Exception as e:
            print(f"  ERROR {sys_name}/{run_name}: {e}")
            continue
        cluster = get_run_cluster(run_name, sys_dir)
        qid = get_run_query_id(run_name, sys_dir)
        query_data[qid] = {
            "urls": {normalize_url(u) for u in cited},
            "cluster": cluster,
        }
    system_cited[sys_name] = query_data
    print(f"  {sys_name}: {len(query_data)} runs")

print()

ALL_SYSTEMS = ["Co-STORM", "STORM", "OmniThink", "OpenAI DR", "Gemini DR"]

# ════════════════════════════════════════════════════════════════════════════
# TABLE 6: Cited UGC prevalence
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TABLE 6: Cited UGC prevalence (tab:cited_ugc)")
print("=" * 80)
print(f"{'System':<15} {'Total Cited':>12} {'UGC Cited':>10} {'UGC %':>8}")
print("-" * 50)

for sys_name in ALL_SYSTEMS:
    total = 0
    ugc = 0
    for qdata in system_cited.get(sys_name, {}).values():
        for url in qdata["urls"]:
            total += 1
            if is_ugc(url):
                ugc += 1
    pct = 100 * ugc / total if total else 0
    print(f"{sys_name:<15} {total:>12,} {ugc:>10,} {pct:>7.1f}%")

print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 2 (CITED): Platform composition of cited UGC
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TABLE 2 (CITED VERSION): Platform composition of cited UGC (% of system's UGC total)")
print("=" * 80)

# Collect per-system platform counts
sys_platform_counts = {}
sys_ugc_totals = {}

for sys_name in ALL_SYSTEMS:
    platform_counts = Counter()
    for qdata in system_cited.get(sys_name, {}).values():
        for url in qdata["urls"]:
            plat = classify_ugc(url)
            if plat:
                platform_counts[plat] += 1
    sys_platform_counts[sys_name] = platform_counts
    sys_ugc_totals[sys_name] = sum(platform_counts.values())

platforms = ["reddit", "youtube", "facebook", "quora", "wikipedia", "medium", "instagram", "tiktok"]

# Print as table
header = f"{'Platform':<15}"
for sys_name in ALL_SYSTEMS:
    header += f" {sys_name:>12}"
print(header)
print("-" * (15 + 13 * len(ALL_SYSTEMS)))

for plat in platforms:
    row = f"{plat + '.com':<15}"
    any_nonzero = False
    for sys_name in ALL_SYSTEMS:
        cnt = sys_platform_counts[sys_name].get(plat, 0)
        total = sys_ugc_totals[sys_name]
        if cnt > 0:
            any_nonzero = True
            pct = 100 * cnt / total if total else 0
            row += f" {pct:>11.1f}%"
        else:
            row += f" {'--':>12}"
    if any_nonzero:
        print(row)

# Other UGC row
row = f"{'Other UGC':<15}"
for sys_name in ALL_SYSTEMS:
    named = sum(sys_platform_counts[sys_name].get(p, 0) for p in platforms)
    total = sys_ugc_totals[sys_name]
    other = total - named
    if other > 0:
        pct = 100 * other / total if total else 0
        row += f" {pct:>11.1f}%"
    else:
        row += f" {'--':>12}"
print(row)

# Raw counts
print(f"\n{'(raw counts)':<15}")
for plat in platforms:
    row = f"{plat:<15}"
    for sys_name in ALL_SYSTEMS:
        cnt = sys_platform_counts[sys_name].get(plat, 0)
        if cnt > 0:
            row += f" {cnt:>12}"
        else:
            row += f" {'--':>12}"
    print(row)

print(f"\n{'TOTAL UGC':<15}", end="")
for sys_name in ALL_SYSTEMS:
    print(f" {sys_ugc_totals[sys_name]:>12}", end="")
print()

print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 (CITED): Cited UGC overlap within topic clusters
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TABLE 3 (CITED VERSION): Cited UGC overlap within topic clusters")
print("=" * 80)

recurring_cited = {}

for sys_name in ALL_SYSTEMS:
    cluster_url_counts = defaultdict(lambda: Counter())
    for qid, qdata in system_cited.get(sys_name, {}).items():
        cluster = qdata["cluster"]
        if not cluster:
            continue
        ugc_urls = {u for u in qdata["urls"] if is_ugc(u)}
        for u in ugc_urls:
            cluster_url_counts[cluster][u] += 1

    recurring = {}
    for cluster, url_counts in cluster_url_counts.items():
        rec = {u: c for u, c in url_counts.items() if c >= 2}
        if rec:
            recurring[cluster] = rec

    recurring_cited[sys_name] = recurring
    total_rec = sum(len(urls) for urls in recurring.values())
    max_freq = max((c for urls in recurring.values() for c in urls.values()), default=0)
    clusters_with_rec = len(recurring)
    total_clusters = df["cluster_id"].nunique()

    print(f"\n{sys_name}:")
    print(f"  Recurring cited UGC URLs:    {total_rec}")
    print(f"  Max single-URL frequency:    {max_freq}")
    print(f"  Clusters with >=1 recurring: {clusters_with_rec}/{total_clusters}")

print()

# LaTeX-ready table
print("LaTeX-ready (Table 3 cited version):")
print(f"{'':30}", end="")
for sys_name in ALL_SYSTEMS:
    print(f" {sys_name:>12}", end="")
print()
print("-" * (30 + 13 * len(ALL_SYSTEMS)))

row_names = ["Recurring UGC URLs", "Max single-URL freq.", "Clusters w/ recurring"]
for row_name in row_names:
    print(f"{row_name:<30}", end="")
    for sys_name in ALL_SYSTEMS:
        rec = recurring_cited.get(sys_name, {})
        if row_name == "Recurring UGC URLs":
            val = sum(len(urls) for urls in rec.values())
        elif row_name == "Max single-URL freq.":
            val = max((c for urls in rec.values() for c in urls.values()), default=0)
        else:
            val = f"{len(rec)}/{df['cluster_id'].nunique()}"
        print(f" {str(val):>12}", end="")
    print()

print()

# ════════════════════════════════════════════════════════════════════════════
# TABLE 4 (CITED): Per-cluster recurring cited UGC
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TABLE 4 (CITED VERSION): Recurring cited UGC URLs per topic cluster")
print("=" * 80)

all_clusters = sorted(df["cluster_id"].unique())
print(f"\n{'Cluster':<35}", end="")
for sys_name in ALL_SYSTEMS:
    print(f" {sys_name:>12}", end="")
print()
print("-" * (35 + 13 * len(ALL_SYSTEMS)))

for cluster in all_clusters:
    print(f"{cluster:<35}", end="")
    for sys_name in ALL_SYSTEMS:
        rec = recurring_cited.get(sys_name, {}).get(cluster, {})
        print(f" {len(rec):>12}", end="")
    print()

print("-" * (35 + 13 * len(ALL_SYSTEMS)))
print(f"{'TOTAL':<35}", end="")
for sys_name in ALL_SYSTEMS:
    total = sum(len(urls) for urls in recurring_cited.get(sys_name, {}).values())
    print(f" {total:>12}", end="")
print()

print()

# ════════════════════════════════════════════════════════════════════════════
# CROSS-SYSTEM OVERLAP (cited URLs)
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("CROSS-SYSTEM OVERLAP (cited recurring URLs)")
print("=" * 80)

recurring_sets = {}
for sys_name in ALL_SYSTEMS:
    pairs = set()
    for cluster, urls in recurring_cited.get(sys_name, {}).items():
        for u in urls:
            pairs.add((cluster, u))
    recurring_sets[sys_name] = pairs
    print(f"  {sys_name}: {len(pairs)} recurring (cluster, url) pairs")

print()
print("Pairwise Jaccard (all 10 pairs):")
print(f"{'Pair':<35} {'|A|':>5} {'|B|':>5} {'|A∩B|':>6} {'|A∪B|':>6} {'Jaccard':>8}")
print("-" * 70)

for i in range(len(ALL_SYSTEMS)):
    for j in range(i + 1, len(ALL_SYSTEMS)):
        a, b = ALL_SYSTEMS[i], ALL_SYSTEMS[j]
        sa, sb = recurring_sets[a], recurring_sets[b]
        inter = sa & sb
        union = sa | sb
        jac = len(inter) / len(union) if union else 0
        print(f"{a + ' / ' + b:<35} {len(sa):>5} {len(sb):>5} {len(inter):>6} {len(union):>6} {jac:>7.3f}")

# URLs recurring in all 5
all_five = recurring_sets["Co-STORM"]
for s in ALL_SYSTEMS[1:]:
    all_five = all_five & recurring_sets[s]
print(f"\nRecurring in ALL 5 systems: {len(all_five)}")
for cluster, u in sorted(all_five):
    print(f"  {cluster}: {u[:80]}")

# URLs recurring in all 3 open-source
open_three = recurring_sets["Co-STORM"] & recurring_sets["STORM"] & recurring_sets["OmniThink"]
print(f"\nRecurring in all 3 open-source: {len(open_three)}")

# At least 2
at_least_two = set()
for i in range(len(ALL_SYSTEMS)):
    for j in range(i + 1, len(ALL_SYSTEMS)):
        at_least_two |= (recurring_sets[ALL_SYSTEMS[i]] & recurring_sets[ALL_SYSTEMS[j]])
print(f"Recurring in at least 2 systems: {len(at_least_two)}")

print()

# ════════════════════════════════════════════════════════════════════════════
# TOP RECURRING CITED UGC URLS per system
# ════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("TOP 10 RECURRING CITED UGC URLs per system")
print("=" * 80)

for sys_name in ALL_SYSTEMS:
    all_rec = []
    for cluster, urls in recurring_cited.get(sys_name, {}).items():
        for u, c in urls.items():
            all_rec.append((c, cluster, u))
    all_rec.sort(reverse=True)
    print(f"\n{sys_name}:")
    for c, cluster, u in all_rec[:10]:
        print(f"  freq={c:>2d}  {cluster:<35s}  {u[:70]}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
