#!/usr/bin/env python3
"""
Batch runner for OpenAI Deep Research over the GEO dataset.

Uses the OpenAI Responses API with o3-deep-research or o4-mini-deep-research.
Since this is a closed-source system, we can only observe the final report
and its cited sources (annotations), not the full retrieval log.

Usage (single test query):
    cd geo_storm
    python -m examples.openai_dr_batch.run_openai_dr_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/openai_dr/clean_runs \
        --limit 1

Usage (full run):
    python -m examples.openai_dr_batch.run_openai_dr_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/openai_dr/clean_runs

Usage (specific clusters only):
    python -m examples.openai_dr_batch.run_openai_dr_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/openai_dr/clean_runs \
        --clusters best_mexican_food comcast_xfinity_cancel
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

from urllib.parse import urlparse, urlunparse

import toml
from openai import OpenAI


def _strip_fragment(url: str) -> str:
    """Strip fragment (#...) from URL to get the base page URL."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _load_secrets():
    """Load API keys from secrets.toml into environment."""
    for p in [
        Path(__file__).resolve().parents[2] / "secrets.toml",
        Path("secrets.toml"),
    ]:
        if p.exists():
            secrets = toml.load(str(p))
            for k, v in secrets.items():
                if k not in os.environ:
                    os.environ[k] = str(v)
            return
    print("Warning: no secrets.toml found", file=sys.stderr)


def run_one(
    *,
    client: OpenAI,
    topic: str,
    run_id: str,
    output_dir: str,
    model: str = "o4-mini-deep-research",
    skip_if_exists: bool = True,
    poll_interval: int = 5,
) -> dict:
    """Run a single OpenAI Deep Research query."""
    run_dir = os.path.join(output_dir, run_id)
    article_path = os.path.join(run_dir, "article.txt")

    # Skip if already completed
    if skip_if_exists and os.path.isfile(article_path):
        return {"run_id": run_id, "status": "skipped"}

    os.makedirs(run_dir, exist_ok=True)

    try:
        # Submit deep research request in background mode
        response = client.responses.create(
            model=model,
            input=topic,
            background=True,
            tools=[
                {"type": "web_search_preview"},
            ],
        )

        response_id = response.id
        print(f"    [{run_id}] Submitted, response_id={response_id}, polling...")

        # Poll until complete (max 30 min to avoid infinite loops)
        max_polls = 1800 // poll_interval  # 30 minutes
        polls = 0
        while response.status in ("queued", "in_progress"):
            polls += 1
            if polls > max_polls:
                _save_raw_response(run_dir, response)
                return {"run_id": run_id, "status": "error", "error": f"Polling timeout after {polls * poll_interval}s"}
            time.sleep(poll_interval)
            response = client.responses.retrieve(response_id)

        if response.status == "failed":
            error_msg = getattr(response, "error", None) or "unknown error"
            # Save raw response for debugging
            _save_raw_response(run_dir, response)
            return {"run_id": run_id, "status": "error", "error": f"API failed: {error_msg}"}

        if response.status == "incomplete":
            _save_raw_response(run_dir, response)
            return {"run_id": run_id, "status": "error", "error": "Response incomplete (possibly timed out)"}

        # Extract the final report text and annotations
        report_text = ""
        annotations = []
        all_search_queries = []

        for item in response.output:
            # Message items contain the final report
            if item.type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text":
                        report_text += content_block.text
                        # Extract annotations (citations with URLs)
                        if hasattr(content_block, "annotations") and content_block.annotations:
                            for ann in content_block.annotations:
                                if hasattr(ann, "url") and ann.url:
                                    annotations.append({
                                        "url": ann.url,
                                        "title": getattr(ann, "title", ""),
                                        "start_index": getattr(ann, "start_index", None),
                                        "end_index": getattr(ann, "end_index", None),
                                    })
            # Track web search calls for analysis
            elif item.type == "web_search_call":
                query_str = getattr(item, "query", "") or getattr(item, "action", {})
                all_search_queries.append(str(query_str))

        # Deduplicate by base URL (strip #:~:text=... fragments)
        seen_base = set()
        unique_sources = []
        for ann in annotations:
            base = _strip_fragment(ann["url"])
            if base not in seen_base:
                seen_base.add(base)
                unique_sources.append({**ann, "base_url": base})

        # Save article text
        with open(article_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Save references (same format as other systems for compatibility)
        # url_to_unified_index uses base URLs (fragment-stripped) for source counting
        ref_data = {
            "url_to_unified_index": {src["base_url"]: i for i, src in enumerate(unique_sources)},
            "url_to_info": {
                src["base_url"]: {
                    "title": src.get("title", ""),
                    "url": src["base_url"],
                    "description": "",
                }
                for src in unique_sources
            },
            "annotations": annotations,  # full list with fragment URLs and positions
            "search_queries": all_search_queries,
        }
        with open(os.path.join(run_dir, "references.json"), "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

        # Save raw response for debugging
        _save_raw_response(run_dir, response)

        return {
            "run_id": run_id,
            "status": "done",
            "cited_sources": len(unique_sources),
            "search_queries": len(all_search_queries),
        }

    except Exception as e:
        traceback.print_exc()
        return {"run_id": run_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def _save_raw_response(run_dir: str, response):
    """Save the raw API response object as JSON for debugging."""
    try:
        raw_path = os.path.join(run_dir, "raw_response.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response.model_dump_json(indent=2))
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Batch OpenAI Deep Research runner for GEO dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="GEO dataset CSV.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--model", type=str, default="o4-mini-deep-research",
                   choices=["o3-deep-research", "o4-mini-deep-research"],
                   help="Deep Research model (default: o4-mini-deep-research).")
    p.add_argument("--limit", type=int, default=0, help="Max queries (0=all).")
    p.add_argument("--clusters", nargs="*", default=None,
                   help="Only run queries from these cluster IDs.")
    p.add_argument("--poll-interval", type=int, default=5,
                   help="Seconds between status polls (default: 5).")
    args = p.parse_args()

    _load_secrets()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=3600,
    )

    with args.dataset.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    for row in rows:
        geo_id = (row.get("geo_id") or "").strip()
        qid = (row.get("question_id") or "").strip()
        cluster = (row.get("cluster_id") or "").strip()
        topic = (row.get("query") or "").strip()
        if not geo_id or not qid or not topic:
            continue
        if args.clusters and cluster not in args.clusters:
            continue
        tasks.append({
            "topic": topic,
            "run_id": f"{geo_id}__{qid}",
        })

    if args.limit > 0:
        tasks = tasks[: args.limit]

    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {len(tasks)} queries -> {output_dir}")
    print(f"Model: {args.model}")

    # Run sequentially (Deep Research is expensive and rate-limited)
    done = 0
    errors = []
    for t in tasks:
        done += 1
        tag = f"[{done}/{len(tasks)}]"
        print(f"  {tag} {t['run_id']}: {t['topic'][:60]}...")
        result = run_one(
            client=client,
            topic=t["topic"],
            run_id=t["run_id"],
            output_dir=output_dir,
            model=args.model,
            poll_interval=args.poll_interval,
        )
        if result["status"] == "error":
            errors.append(result)
            print(f"  {tag} {result['run_id']} FAILED: {result['error']}")
        else:
            extra = ""
            if "cited_sources" in result:
                extra = f" (sources={result['cited_sources']}, searches={result['search_queries']})"
            print(f"  {tag} {result['run_id']} -> {result['status']}{extra}")

    print(f"\nDone: {done}, Errors: {len(errors)}")
    for e in errors:
        print(f"  {e['run_id']}: {e['error']}")


if __name__ == "__main__":
    main()
