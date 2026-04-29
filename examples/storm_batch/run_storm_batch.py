#!/usr/bin/env python3
"""
Batch runner for STORM Wiki pipeline over the GEO dataset.

Uses gpt-4o-mini (same as Co-STORM demo preset) + Serper retrieval.
Optionally wraps retriever with UGCMimicRetriever for adversarial injection.

Usage (clean baseline):
    PYTHONPATH=. python -m examples.storm_batch.run_storm_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/storm_clean_runs \
        --workers 4

Usage (UGC mimic):
    PYTHONPATH=. python -m examples.storm_batch.run_storm_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/storm_ugc_1url_runs \
        --ugc-config geo_out/storm_ugc_config_1url.json \
        --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import SerperRM
from knowledge_storm.utils import load_api_key
from examples.batch._injector import UGCMimicRetriever


def _build_lm_configs() -> STORMWikiLMConfigs:
    openai_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }
    model = "gpt-4o-mini"
    lm = STORMWikiLMConfigs()
    lm.set_conv_simulator_lm(OpenAIModel(model=model, max_tokens=500, **openai_kwargs))
    lm.set_question_asker_lm(OpenAIModel(model=model, max_tokens=500, **openai_kwargs))
    lm.set_outline_gen_lm(OpenAIModel(model=model, max_tokens=400, **openai_kwargs))
    lm.set_article_gen_lm(OpenAIModel(model=model, max_tokens=700, **openai_kwargs))
    lm.set_article_polish_lm(OpenAIModel(model=model, max_tokens=4000, **openai_kwargs))
    return lm


def _build_rm(
    ugc_rule: Optional[dict] = None,
    ugc_append_mode: bool = False,
    enable_arctic_shift: bool = False,
    merge_snippets: bool = False,
) -> SerperRM:
    rm = SerperRM(
        serper_search_api_key=os.getenv("SERPER_API_KEY"),
        query_params={"autocorrect": True, "num": 10, "page": 1},
        ENABLE_EXTRA_SNIPPET_EXTRACTION=True,
        enable_arctic_shift=enable_arctic_shift,
        merge_snippets=merge_snippets,
    )
    if ugc_rule:
        rm = UGCMimicRetriever(
            base_retriever=rm,
            target_url=ugc_rule.get("target_url", ""),
            target_urls=ugc_rule.get("target_urls", []),
            domain_prefixes=ugc_rule.get("domain_prefixes", []),
            adversarial_text=ugc_rule.get("adversarial_text", ""),
            separator=ugc_rule.get("separator", "--- Additional comment excerpt ---"),
            append_mode=ugc_append_mode,
        )
    return rm


def _load_ugc_rule(config_path: str, question_id: str) -> Optional[dict]:
    """Load per-question UGC rule from config JSON."""
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    rules = config.get("rules_by_question_id", {})
    separator = config.get("separator", "--- Additional comment excerpt ---")
    rule = rules.get(question_id)
    if not rule:
        return None
    rule["separator"] = separator
    return rule


def run_one(
    *,
    topic: str,
    run_id: str,
    question_id: str,
    output_dir: str,
    ugc_config_path: str = "",
    ugc_append_mode: bool = False,
    enable_arctic_shift: bool = False,
    merge_snippets: bool = False,
    skip_if_exists: bool = True,
) -> dict:
    """Run a single STORM query and rename output to run_id directory."""
    run_dir = os.path.join(output_dir, run_id)
    article_path = os.path.join(run_dir, "storm_gen_article.txt")

    if skip_if_exists and os.path.exists(article_path):
        return {"run_id": run_id, "status": "skipped"}

    storm_dir = None
    try:
        load_api_key(toml_file_path="secrets.toml")
        lm_configs = _build_lm_configs()
        ugc_rule = _load_ugc_rule(ugc_config_path, question_id)
        rm = _build_rm(ugc_rule, ugc_append_mode=ugc_append_mode,
                        enable_arctic_shift=enable_arctic_shift,
                        merge_snippets=merge_snippets)

        engine_args = STORMWikiRunnerArguments(
            output_dir=output_dir,
            max_conv_turn=3,
            max_perspective=3,
            search_top_k=3,
            max_thread_num=3,
        )

        runner = STORMWikiRunner(engine_args, lm_configs, rm)
        runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
        runner.post_run()
        storm_dir = runner.article_output_dir

        # STORM saves to output_dir/<topic_name>/. Move to output_dir/<run_id>/.
        if os.path.exists(storm_dir) and os.path.normpath(storm_dir) != os.path.normpath(run_dir):
            os.makedirs(run_dir, exist_ok=True)
            for f in os.listdir(storm_dir):
                shutil.move(os.path.join(storm_dir, f), os.path.join(run_dir, f))
            shutil.rmtree(storm_dir, ignore_errors=True)

        # Write UGC mimic metadata if injection was used
        if ugc_rule and isinstance(rm, UGCMimicRetriever):
            meta = {
                "question_id": question_id,
                "target_urls": [ugc_rule.get("target_url", "")] if ugc_rule.get("target_url") else ugc_rule.get("target_urls", []),
                "domain_prefixes": ugc_rule.get("domain_prefixes", []),
                "ugc_exposure": rm.patched_count > 0,
                "ugc_exposure_count": rm.patched_count,
                "matched_urls": rm.matched_urls,
                "separator": ugc_rule.get("separator", ""),
            }
            meta_path = os.path.join(run_dir, "ugc_mimic_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        return {"run_id": run_id, "status": "done"}

    except Exception as e:
        traceback.print_exc()
        if storm_dir and os.path.exists(storm_dir) and os.path.normpath(storm_dir) != os.path.normpath(run_dir):
            shutil.rmtree(storm_dir, ignore_errors=True)
        return {"run_id": run_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser(description="Batch STORM runner for GEO dataset.")
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

    # If UGC config provided, load eligible question_ids
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
        geo_id = (row.get("geo_id") or "").strip()
        qid = (row.get("question_id") or "").strip()
        topic = (row.get("query") or "").strip()
        if not geo_id or not qid or not topic:
            continue
        if eligible_qids is not None and qid not in eligible_qids:
            continue
        tasks.append({"topic": topic, "run_id": f"{geo_id}__{qid}", "question_id": qid})

    if args.limit > 0:
        tasks = tasks[:args.limit]

    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {len(tasks)} queries with {args.workers} workers -> {output_dir}")

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                topic=t["topic"],
                run_id=t["run_id"],
                question_id=t["question_id"],
                output_dir=output_dir,
                ugc_config_path=args.ugc_config,
                ugc_append_mode=args.ugc_append_mode,
                enable_arctic_shift=args.enable_arctic_shift,
                merge_snippets=args.merge_snippets,
            ): t
            for t in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1
            tag = f"[{done}/{len(tasks)}]"
            if result["status"] == "error":
                errors.append(result)
                print(f"  {tag} {result['run_id']} FAILED: {result['error']}")
            else:
                print(f"  {tag} {result['run_id']} -> {result['status']}")

    print(f"\nDone: {done}, Errors: {len(errors)}")
    for e in errors:
        print(f"  {e['run_id']}: {e['error']}")


if __name__ == "__main__":
    main()
