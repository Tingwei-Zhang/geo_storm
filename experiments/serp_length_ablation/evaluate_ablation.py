#!/usr/bin/env python3
"""
Step 3: Evaluate all length variants and produce a summary table.

Usage:
    cd geo_storm
    conda run -n storm python -m experiments.serp_length_ablation.evaluate_ablation
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_ugc import evaluate_run, TARGET_NAMES

CLUSTER = "comcast_xfinity_cancel"
TARGET_NAME = TARGET_NAMES[CLUSTER]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
VARIANTS = ["10", "15", "20", "25", "30", "40", "original"]


def evaluate_variant(variant: str) -> dict:
    variant_dir = RUNS_DIR / variant
    if not variant_dir.exists():
        return {"variant": variant, "total": 0, "error": "no runs"}

    run_dirs = [d for d in variant_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    total = 0
    exposed = 0
    cited = 0
    mentioned = 0
    cited_exposed = 0
    mentioned_exposed = 0

    for rd in sorted(run_dirs):
        if rd.name in ("__pycache__",):
            continue
        r = evaluate_run(rd, TARGET_NAME)
        if not r["exists"]:
            continue
        total += 1
        if r["ugc_exposure"]:
            exposed += 1
            if r["cited"]:
                cited_exposed += 1
            if r["mentioned"]:
                mentioned_exposed += 1
        if r["cited"]:
            cited += 1
        if r["mentioned"]:
            mentioned += 1

    return {
        "variant": variant,
        "total": total,
        "exposed": exposed,
        "exposure_rate": exposed / total if total else 0,
        "cited": cited,
        "cite_rate": cited / total if total else 0,
        "mentioned": mentioned,
        "mention_rate": mentioned / total if total else 0,
        "cited_if_exposed": cited_exposed,
        "cite_rate_if_exposed": cited_exposed / exposed if exposed else 0,
        "mentioned_if_exposed": mentioned_exposed,
        "mention_rate_if_exposed": mentioned_exposed / exposed if exposed else 0,
    }


def main():
    print(f"Evaluating SERP length ablation for {CLUSTER} (target: {TARGET_NAME})")
    print(f"Runs dir: {RUNS_DIR}\n")

    results = []
    for v in VARIANTS:
        r = evaluate_variant(v)
        results.append(r)

    # Print table
    header = f"{'Variant':>10} {'Total':>6} {'Exposed':>8} {'Exp%':>7} {'Cited':>6} {'Cite%':>7} {'Ment':>6} {'Ment%':>7} {'Cite|Exp':>9} {'Ment|Exp':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r.get("error"):
            print(f"{r['variant']:>10} -- {r['error']}")
            continue
        print(
            f"{r['variant']:>10} {r['total']:>6} {r['exposed']:>8} "
            f"{r['exposure_rate']:>6.1%} {r['cited']:>6} {r['cite_rate']:>6.1%} "
            f"{r['mentioned']:>6} {r['mention_rate']:>6.1%} "
            f"{r['cite_rate_if_exposed']:>8.1%} {r['mention_rate_if_exposed']:>8.1%}"
        )

    output_path = RUNS_DIR / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
