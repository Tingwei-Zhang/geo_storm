#!/usr/bin/env python3
"""
Evaluate all three systems' 13-word SERP attack results.
Reports overall and per-cluster metrics.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_ugc import evaluate_run, TARGET_NAMES

RUNS_DIR = Path(__file__).resolve().parent / "runs"

CLUSTER_ORDER = [
    "comcast_xfinity_cancel",
    "dating_apps_divorced_men_over_50",
    "antivirus_software",
    "best_mexican_food",
    "best_brunch",
    "product_comparison",
    "amazon_prime_cancel",
    "supplements_building_muscle",
    "aaa_alternative",
    "crypto_invest",
    "supplements_weight_loss",
]


def get_cluster(question_id: str) -> str:
    for cluster in sorted(CLUSTER_ORDER, key=len, reverse=True):
        if question_id.startswith(cluster):
            return cluster
    return "unknown"


def evaluate_system(system: str) -> dict:
    sys_dir = RUNS_DIR / system
    if not sys_dir.exists():
        return {}

    run_dirs = [d for d in sys_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]

    per_cluster = defaultdict(lambda: {
        "total": 0, "exposed": 0, "cited": 0, "mentioned": 0,
        "cited_exposed": 0, "mentioned_exposed": 0,
    })
    overall = {
        "total": 0, "exposed": 0, "cited": 0, "mentioned": 0,
        "cited_exposed": 0, "mentioned_exposed": 0,
    }

    for rd in sorted(run_dirs):
        # Extract question_id from run_id (geo_id__question_id)
        run_name = rd.name
        if "__" in run_name:
            qid = run_name.split("__", 1)[1]
        else:
            qid = run_name

        cluster = get_cluster(qid)
        target_name = TARGET_NAMES.get(cluster, "")
        if not target_name:
            continue

        r = evaluate_run(rd, target_name)
        if not r["exists"]:
            continue

        for bucket in [overall, per_cluster[cluster]]:
            bucket["total"] += 1
            if r["ugc_exposure"]:
                bucket["exposed"] += 1
                if r["cited"]:
                    bucket["cited_exposed"] += 1
                if r["mentioned"]:
                    bucket["mentioned_exposed"] += 1
            if r["cited"]:
                bucket["cited"] += 1
            if r["mentioned"]:
                bucket["mentioned"] += 1

    return {"overall": overall, "per_cluster": dict(per_cluster)}


def fmt_pct(num, denom):
    if denom == 0:
        return "--"
    return f"{100 * num / denom:.1f}"


def print_results(system: str, data: dict):
    o = data["overall"]
    print(f"\n{'='*70}")
    print(f"{system.upper()} — Overall ({o['total']} queries)")
    print(f"{'='*70}")
    print(f"  Exposure:  {o['exposed']}/{o['total']} = {fmt_pct(o['exposed'], o['total'])}%")
    print(f"  Cited:     {o['cited']}/{o['total']} = {fmt_pct(o['cited'], o['total'])}%")
    print(f"  Mentioned: {o['mentioned']}/{o['total']} = {fmt_pct(o['mentioned'], o['total'])}%")
    print(f"  Cited|Exp: {fmt_pct(o['cited_exposed'], o['exposed'])}%")
    print(f"  Ment|Exp:  {fmt_pct(o['mentioned_exposed'], o['exposed'])}%")

    print(f"\n  Per-cluster:")
    print(f"  {'Cluster':<30} {'N':>4} {'Exp%':>6} {'Cite%':>6} {'Ment%':>6} {'C|E':>6} {'M|E':>6}")
    print(f"  {'-'*66}")
    for cluster in CLUSTER_ORDER:
        c = data["per_cluster"].get(cluster, {})
        if not c or c["total"] == 0:
            continue
        print(f"  {cluster:<30} {c['total']:>4} {fmt_pct(c['exposed'], c['total']):>6} "
              f"{fmt_pct(c['cited'], c['total']):>6} {fmt_pct(c['mentioned'], c['total']):>6} "
              f"{fmt_pct(c['cited_exposed'], c['exposed']):>6} {fmt_pct(c['mentioned_exposed'], c['exposed']):>6}")


def main():
    all_results = {}
    for system in ["costorm", "storm", "omnithink"]:
        data = evaluate_system(system)
        if data:
            all_results[system] = data
            print_results(system, data)

    out_path = RUNS_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
