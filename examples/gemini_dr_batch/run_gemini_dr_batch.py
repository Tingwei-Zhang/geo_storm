#!/usr/bin/env python3
"""
Batch runner for Gemini Deep Research over the GEO dataset.

Uses the Google GenAI Interactions API with deep-research-pro-preview-12-2025.
Since this is a closed-source system, we can only observe the final report
and its cited sources (grounding metadata), not the full retrieval log.

Requires: pip install google-genai
Requires: GOOGLE_API_KEY or GEMINI_API_KEY environment variable

Usage (single test query):
    cd geo_storm
    python -m examples.gemini_dr_batch.run_gemini_dr_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/gemini_dr/clean_runs \
        --limit 1

Usage (specific clusters):
    python -m examples.gemini_dr_batch.run_gemini_dr_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/gemini_dr/clean_runs \
        --clusters best_mexican_food comcast_xfinity_cancel
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import toml
from google import genai


AGENT_MODEL = "deep-research-pro-preview-12-2025"


def _strip_fragment(url: str) -> str:
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


def _extract_urls_from_markdown(text: str) -> list[dict]:
    """Extract URLs from markdown-style citations in the report text.

    Gemini Deep Research embeds citations as markdown links like:
        [texasstandard.org](https://vertexaisearch.cloud.google.com/grounding-api-redirect/...)

    The proxy URLs don't resolve externally, so we reconstruct a usable URL
    from the title (which contains the actual domain, e.g. "reddit.com").
    """
    pattern = r'\[([^\]]*)\]\((https?://[^\)]+)\)'
    urls = []
    seen = set()
    for match in re.finditer(pattern, text):
        title = match.group(1).strip()
        proxy_url = match.group(2)

        # The title IS the actual domain for Gemini proxy URLs
        # Reconstruct a usable URL: https://<domain>/
        if "vertexaisearch.cloud.google.com/grounding-api-redirect" in proxy_url:
            if title and "." in title:
                resolved = f"https://{title}"
            else:
                resolved = proxy_url  # fallback
        else:
            resolved = _strip_fragment(proxy_url)

        if resolved not in seen:
            seen.add(resolved)
            urls.append({"url": resolved, "title": title, "proxy_url": proxy_url})
    return urls


def _extract_grounding_urls(interaction) -> list[dict]:
    """Extract URLs from grounding metadata in the interaction outputs."""
    urls = []
    seen = set()

    for output in (interaction.outputs or []):
        # Check if the output has grounding_metadata
        grounding = getattr(output, "grounding_metadata", None)
        if grounding is None:
            # Try nested: output might be a Content with candidates
            candidates = getattr(output, "candidates", None)
            if candidates:
                for cand in candidates:
                    grounding = getattr(cand, "grounding_metadata", None)
                    if grounding:
                        _process_grounding(grounding, urls, seen)
            continue
        _process_grounding(grounding, urls, seen)

    return urls


def _process_grounding(grounding, urls: list, seen: set):
    """Process a GroundingMetadata object to extract web URLs."""
    chunks = getattr(grounding, "grounding_chunks", None) or []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web:
            uri = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or ""
            if uri:
                base = _strip_fragment(uri)
                if base not in seen:
                    seen.add(base)
                    urls.append({"url": base, "title": title, "raw_url": uri})


def run_one(
    *,
    client: genai.Client,
    topic: str,
    run_id: str,
    output_dir: str,
    skip_if_exists: bool = True,
    poll_interval: int = 10,
) -> dict:
    """Run a single Gemini Deep Research query."""
    run_dir = os.path.join(output_dir, run_id)
    article_path = os.path.join(run_dir, "article.txt")

    if skip_if_exists and os.path.isfile(article_path):
        return {"run_id": run_id, "status": "skipped"}

    os.makedirs(run_dir, exist_ok=True)

    try:
        # Submit deep research request
        interaction = client.interactions.create(
            input=topic,
            agent=AGENT_MODEL,
            background=True,
        )

        interaction_id = interaction.id
        print(f"    [{run_id}] Submitted, interaction_id={interaction_id}, polling...")

        # Poll until complete (max 30 min to avoid infinite loops)
        max_polls = 1800 // poll_interval  # 30 minutes
        polls = 0
        while interaction.status not in ("completed", "failed"):
            polls += 1
            if polls > max_polls:
                _save_raw_interaction(run_dir, interaction)
                return {"run_id": run_id, "status": "error", "error": f"Polling timeout after {polls * poll_interval}s"}
            time.sleep(poll_interval)
            interaction = client.interactions.get(interaction_id)

        if interaction.status == "failed":
            _save_raw_interaction(run_dir, interaction)
            return {"run_id": run_id, "status": "error", "error": "Gemini DR failed"}

        # Extract report text from the last output
        report_text = ""
        if interaction.outputs:
            last_output = interaction.outputs[-1]
            report_text = getattr(last_output, "text", "") or ""

        # Extract cited URLs from multiple sources
        # 1. Grounding metadata (structured)
        grounding_urls = _extract_grounding_urls(interaction)
        # 2. Markdown links in the text (fallback / supplement)
        markdown_urls = _extract_urls_from_markdown(report_text)

        # Merge: grounding URLs take priority, then add any new ones from markdown
        all_urls = list(grounding_urls)
        seen_bases = {u["url"] for u in all_urls}
        for mu in markdown_urls:
            if mu["url"] not in seen_bases:
                all_urls.append(mu)
                seen_bases.add(mu["url"])

        # Save article text
        with open(article_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # Save references (compatible format)
        ref_data = {
            "url_to_unified_index": {u["url"]: i for i, u in enumerate(all_urls)},
            "url_to_info": {
                u["url"]: {
                    "title": u.get("title", ""),
                    "url": u["url"],
                    "description": "",
                }
                for u in all_urls
            },
            "grounding_urls": grounding_urls,
            "markdown_urls": markdown_urls,
        }
        with open(os.path.join(run_dir, "references.json"), "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

        # Save raw interaction for debugging
        _save_raw_interaction(run_dir, interaction)

        return {
            "run_id": run_id,
            "status": "done",
            "cited_sources": len(all_urls),
            "grounding_sources": len(grounding_urls),
            "markdown_sources": len(markdown_urls),
        }

    except Exception as e:
        traceback.print_exc()
        return {"run_id": run_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def _save_raw_interaction(run_dir: str, interaction):
    """Save the raw interaction object for debugging."""
    try:
        raw_path = os.path.join(run_dir, "raw_interaction.json")
        # Try model_dump_json first, fall back to str
        if hasattr(interaction, "model_dump_json"):
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(interaction.model_dump_json(indent=2))
        elif hasattr(interaction, "to_json_string"):
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(interaction.to_json_string())
        else:
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(str(interaction))
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Batch Gemini Deep Research runner for GEO dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="GEO dataset CSV.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--limit", type=int, default=0, help="Max queries (0=all).")
    p.add_argument("--clusters", nargs="*", default=None,
                   help="Only run queries from these cluster IDs.")
    p.add_argument("--poll-interval", type=int, default=10,
                   help="Seconds between status polls (default: 10).")
    args = p.parse_args()

    _load_secrets()

    # The google-genai SDK reads GOOGLE_API_KEY or GEMINI_API_KEY from env
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

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
    print(f"Model: {AGENT_MODEL}")

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
            poll_interval=args.poll_interval,
        )
        if result["status"] == "error":
            errors.append(result)
            print(f"  {tag} {result['run_id']} FAILED: {result['error']}")
        else:
            extra = ""
            if "cited_sources" in result:
                extra = f" (sources={result['cited_sources']}, grounding={result['grounding_sources']}, markdown={result['markdown_sources']})"
            print(f"  {tag} {result['run_id']} -> {result['status']}{extra}")

    print(f"\nDone: {done}, Errors: {len(errors)}")
    for e in errors:
        print(f"  {e['run_id']}: {e['error']}")


if __name__ == "__main__":
    main()
