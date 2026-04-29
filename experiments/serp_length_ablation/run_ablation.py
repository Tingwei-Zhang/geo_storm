#!/usr/bin/env python3
"""
Step 2: Run Co-STORM for each length variant on the comcast_xfinity_cancel cluster.

For each compressed config (10w, 15w, 20w, 25w, 30w, 40w, original), runs Co-STORM
with ugc_append_mode=True on all comcast queries.

Usage:
    cd geo_storm
    conda run -n storm python -m experiments.serp_length_ablation.run_ablation [--lengths 10 15 20 25 30 40 original]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.batch.run_single_query import run_single_query

CLUSTER = "comcast_xfinity_cancel"
DATASET_CSV = PROJECT_ROOT / "geo_out" / "geo_dataset_ugc_aware.csv"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_comcast_queries() -> list[dict]:
    rows = []
    with open(DATASET_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster = (row.get("cluster_id") or "").strip()
            if cluster != CLUSTER:
                continue
            geo_id = (row.get(" geo_id") or row.get("geo_id") or "").strip()
            qid = (row.get("question_id") or "").strip()
            query = (row.get("query") or "").strip()
            if geo_id and qid and query:
                rows.append({"geo_id": geo_id, "question_id": qid, "query": query})
    return rows


def run_variant(variant: str, queries: list[dict], no_skip: bool = False):
    if variant == "original":
        config_path = CONFIGS_DIR / "ugc_config_comcast_original.json"
    else:
        config_path = CONFIGS_DIR / f"ugc_config_comcast_{variant}w.json"

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return

    output_dir = RUNS_DIR / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running variant: {variant} ({config_path.name})")
    print(f"Output: {output_dir}")
    print(f"Queries: {len(queries)}")
    print(f"{'='*60}\n")

    success = 0
    failed = []

    for q in queries:
        run_id = f"{q['geo_id']}__{q['question_id']}"
        print(f"  [{success + len(failed) + 1}/{len(queries)}] {run_id} ...", end=" ", flush=True)
        try:
            ok = run_single_query(
                question_id=run_id,
                topic=q["query"],
                output_dir=output_dir,
                retriever="serper",
                demo_turns=2,
                retrieve_top_k=3,
                total_conv_turn=20,
                max_search_queries=2,
                max_search_thread=5,
                max_search_queries_per_turn=3,
                warmstart_max_num_experts=1,
                warmstart_max_turn_per_experts=1,
                warmstart_max_thread=1,
                max_thread_num=5,
                max_num_round_table_experts=1,
                moderator_override_N_consecutive_answering_turn=2,
                node_expansion_trigger_count=10,
                lm_preset="demo",
                skip_if_exists=not no_skip,
                ugc_mimic_config_path=str(config_path),
                ugc_append_mode=True,
            )
            if ok:
                success += 1
                print("OK")
            else:
                failed.append(run_id)
                print("FAILED")
        except Exception as e:
            failed.append(run_id)
            print(f"ERROR: {e}")

    summary = {"variant": variant, "total": len(queries), "success": success, "failed": failed}
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Done: {success}/{len(queries)} succeeded")
    if failed:
        print(f"  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        nargs="+",
        default=["10", "15", "20", "25", "30", "40", "original"],
        help="Length variants to run (e.g., 10 15 20 25 30 40 original)",
    )
    parser.add_argument("--no-skip", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    queries = load_comcast_queries()
    print(f"Loaded {len(queries)} queries for {CLUSTER}")

    for variant in args.lengths:
        run_variant(variant, queries, no_skip=args.no_skip)

    print("\n\nAll variants complete. Run evaluate_ablation.py to compute metrics.")


if __name__ == "__main__":
    main()
