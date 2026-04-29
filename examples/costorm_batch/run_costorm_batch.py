#!/usr/bin/env python3
"""
Batch runner for Co-STORM pipeline over the GEO dataset.

Uses gpt-4o-mini + Serper retrieval. Optionally wraps retriever with
UGCMimicRetriever for adversarial injection.

Usage (clean baseline):
    python -m examples.costorm_batch.run_costorm_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/costorm/clean_runs \
        --workers 4

Usage (UGC mimic, 1-URL):
    python -m examples.costorm_batch.run_costorm_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir experiments/serp_main_13w/runs/costorm_1url \
        --ugc-config experiments/serp_main_13w/configs/ugc_config_costorm_1url_15w.json \
        --ugc-append-mode \
        --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.batch.run_single_query import run_single_query


def main():
    p = argparse.ArgumentParser(description="Batch Co-STORM runner for GEO dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="GEO dataset CSV.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--ugc-config", type=str, default="", help="UGC mimic config JSON (optional).")
    p.add_argument("--ugc-append-mode", action="store_true",
                   help="Append adversarial text seamlessly (default: off).")
    p.add_argument("--enable-arctic-shift", action="store_true",
                   help="Fetch full Reddit content via Arctic Shift (for full-content attack).")
    p.add_argument("--merge-snippets", action="store_true",
                   help="Merge Arctic Shift content with SERP snippets.")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    p.add_argument("--limit", type=int, default=0, help="Max queries (0=all).")
    args = p.parse_args()

    eligible_qids = None
    if args.ugc_config:
        with open(args.ugc_config, "r", encoding="utf-8") as f:
            ugc_data = json.load(f)
        eligible_qids = set(ugc_data.get("rules_by_question_id", {}).keys())
        print(f"UGC config: {args.ugc_config} ({len(eligible_qids)} eligible)")

    with args.dataset.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    for row in rows:
        geo_id = (row.get(" geo_id") or row.get("geo_id") or "").strip()
        qid = (row.get("question_id") or "").strip()
        topic = (row.get("query") or "").strip()
        if not geo_id or not qid or not topic:
            continue
        if eligible_qids is not None and qid not in eligible_qids:
            continue
        tasks.append({"topic": topic, "run_id": f"{geo_id}__{qid}", "question_id": qid})

    if args.limit > 0:
        tasks = tasks[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(tasks)} queries with {args.workers} workers -> {args.output_dir}")

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_single_query,
                question_id=t["run_id"],
                topic=t["topic"],
                output_dir=args.output_dir,
                retriever="serper",
                lm_preset="demo",
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
                skip_if_exists=True,
                ugc_mimic_config_path=args.ugc_config or None,
                ugc_append_mode=args.ugc_append_mode,
                enable_arctic_shift=args.enable_arctic_shift,
                merge_snippets=args.merge_snippets,
            ): t
            for t in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            t = futures[future]
            tag = f"[{done}/{len(tasks)}]"
            if isinstance(result, bool):
                status = "done" if result else "failed"
                print(f"  {tag} {t['run_id']} -> {status}")
            else:
                print(f"  {tag} {t['run_id']} -> {result}")

    print(f"\nDone: {done}, Errors: {len(errors)}")


if __name__ == "__main__":
    main()
