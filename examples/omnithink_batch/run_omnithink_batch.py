#!/usr/bin/env python3
"""
Batch runner for OmniThink pipeline over the GEO dataset.

Uses gpt-4o-mini + Serper API (default, same as STORM/Co-STORM experiments).
Runs the full OmniThink pipeline: MindMap -> Outline -> Article -> Polish.

Usage (clean baseline):
    PYTHONPATH=omnithink:. python -m examples.omnithink_batch.run_omnithink_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/omnithink/clean_runs \
        --workers 2

Usage (UGC mimic 1-URL):
    PYTHONPATH=omnithink:. python -m examples.omnithink_batch.run_omnithink_batch \
        --dataset geo_out/geo_dataset_clean.csv \
        --output-dir geo_out/omnithink/ugc_1url_runs \
        --ugc-config geo_out/omnithink/ugc_config_1url.json \
        --workers 2

Use --retriever google to switch back to Google Custom Search API.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add OmniThink to sys.path (expects PYTHONPATH=omnithink:. or omnithink/ as sibling)
_omnithink_root = Path(__file__).resolve().parents[2] / "omnithink"
if str(_omnithink_root) not in sys.path:
    sys.path.insert(0, str(_omnithink_root))

# Also ensure project root is on path (for _injector later)
_storm_geo_root = Path(__file__).resolve().parents[2]
if str(_storm_geo_root) not in sys.path:
    sys.path.insert(0, str(_storm_geo_root))

from src.utils.utils import load_api_key
from src.tools.rm import GoogleSearch, SerperSearch
from src.tools.lm import OpenAIModel
from src.tools.mindmap import MindMap
from src.actions.outline_generation import OutlineGenerationModule
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule
from src.dataclass.Article import Article

from examples.omnithink_batch._ugc_injector import UGCMimicRetriever


def _load_secrets():
    """Load API keys from secrets.toml."""
    for p in [
        _storm_geo_root / "secrets.toml",
        _omnithink_root / "secrets.toml",
        _omnithink_root.parent / "secrets.toml",
    ]:
        if p.exists():
            load_api_key(str(p))
            return
    print("Warning: no secrets.toml found", file=sys.stderr)


def run_one(
    *,
    topic: str,
    run_id: str,
    output_dir: str,
    llm_model: str = "gpt-4o-mini",
    retriever_type: str = "serper",
    ugc_rule: dict | None = None,
    ugc_separator: str = "--- Additional comment excerpt ---",
    ugc_append_mode: bool = False,
    enable_arctic_shift: bool = False,
    merge_snippets: bool = False,
    depth: int = 2,
    retriever_k: int = 5,
    retrieve_top_k: int = 3,
    max_thread_num: int = 10,
    skip_if_exists: bool = True,
) -> dict:
    """Run a single OmniThink query, optionally with UGC injection."""
    run_dir = os.path.join(output_dir, run_id)
    article_dir = os.path.join(run_dir, "article")

    # Skip if article already generated
    if skip_if_exists and os.path.isdir(article_dir):
        existing = os.listdir(article_dir)
        if existing:
            return {"run_id": run_id, "status": "skipped"}

    try:
        # Build retriever and LM fresh per run (thread safety)
        if retriever_type == "serper":
            retriever = SerperSearch(
                k=retriever_k,
                enable_arctic_shift=enable_arctic_shift,
                merge_snippets=merge_snippets,
            )
        else:
            retriever = GoogleSearch(k=retriever_k)

        # Wrap with UGC mimic if rule provided
        ugc_retriever = None
        if ugc_rule:
            ugc_retriever = UGCMimicRetriever(
                base_retriever=retriever,
                target_url=ugc_rule.get("target_url", ""),
                target_urls=ugc_rule.get("target_urls"),
                domain_prefixes=ugc_rule.get("domain_prefixes"),
                adversarial_text=ugc_rule["adversarial_text"],
                separator=ugc_separator,
                append_mode=ugc_append_mode,
            )
            retriever = ugc_retriever

        lm_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 1.0,
            "top_p": 0.9,
        }
        lm = OpenAIModel(model=llm_model, max_tokens=2000, **lm_kwargs)

        # Phase 1: Build MindMap (external retrieval)
        mind_map = MindMap(
            retriever=retriever,
            gen_concept_lm=lm,
            depth=depth,
        )
        generator = mind_map.build_map(topic)
        for layer in generator:
            pass  # consume generator

        # Phase 2: Generate outline
        ogm = OutlineGenerationModule(lm)
        outline = ogm.generate_outline(topic=topic, mindmap=mind_map)

        # Phase 3: Generate article (internal retrieval via cosine similarity)
        article_with_outline = Article.from_outline_str(topic=topic, outline_str=outline)
        ag = ArticleGenerationModule(
            retriever=retriever,
            article_gen_lm=lm,
            retrieve_top_k=retrieve_top_k,
            max_thread_num=max_thread_num,
        )
        article = ag.generate_article(
            topic=topic, mindmap=mind_map, article_with_outline=article_with_outline
        )

        # Phase 4: Polish article
        ap = ArticlePolishingModule(article_gen_lm=lm, article_polish_lm=lm)
        article = ap.polish_article(topic=topic, draft_article=article)

        # Save outputs
        for subdir in ["map", "outline", "article"]:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        topic_file = topic.replace(" ", "_")

        # Save MindMap
        mind_map.save_map(mind_map.root, os.path.join(run_dir, "map", topic_file))

        # Save outline
        with open(os.path.join(run_dir, "outline", topic_file), "w", encoding="utf-8") as f:
            f.write(outline)

        # Save article
        with open(os.path.join(run_dir, "article", topic_file), "w", encoding="utf-8") as f:
            f.write(article.to_string())

        # Save references (url_to_info for UGC analysis)
        ref_data = {
            "url_to_unified_index": article.reference.get("url_to_unified_index", {}),
            "url_to_info": {},
        }
        for url, info in article.reference.get("url_to_info", {}).items():
            if isinstance(info, dict):
                ref_data["url_to_info"][url] = info
            else:
                ref_data["url_to_info"][url] = str(info)
        with open(os.path.join(run_dir, "references.json"), "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

        # Save UGC mimic metadata if applicable
        if ugc_retriever is not None:
            ugc_meta = {
                "patched_count": ugc_retriever.patched_count,
                "matched_urls": ugc_retriever.matched_urls,
                "target_url": ugc_rule.get("target_url", ""),
                "target_urls": ugc_rule.get("target_urls"),
                "domain_prefixes": ugc_rule.get("domain_prefixes"),
            }
            with open(os.path.join(run_dir, "ugc_mimic_meta.json"), "w", encoding="utf-8") as f:
                json.dump(ugc_meta, f, ensure_ascii=False, indent=2)

        return {"run_id": run_id, "status": "done"}

    except Exception as e:
        traceback.print_exc()
        return {"run_id": run_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def main():
    p = argparse.ArgumentParser(description="Batch OmniThink runner for GEO dataset.")
    p.add_argument("--dataset", type=Path, required=True, help="GEO dataset CSV.")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--ugc-config", type=Path, default=None, help="UGC mimic config JSON.")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers (default 2).")
    p.add_argument("--limit", type=int, default=0, help="Max queries (0=all).")
    p.add_argument("--depth", type=int, default=2, help="MindMap depth (default 2).")
    p.add_argument("--retriever", type=str, default="serper", choices=["serper", "google"],
                   help="Search API to use (default: serper).")
    p.add_argument("--retriever-k", type=int, default=5, help="Search results per query.")
    p.add_argument("--retrieve-top-k", type=int, default=3, help="Internal retrieval top-k.")
    p.add_argument("--llm", type=str, default="gpt-4o-mini", help="LLM model name (default: gpt-4o-mini).")
    p.add_argument("--enable-arctic-shift", action="store_true",
                   help="Use Arctic Shift API for full Reddit thread content (default: off).")
    p.add_argument("--merge-snippets", action="store_true",
                   help="Merge SERP snippet + page content into one snippet (default: off).")
    p.add_argument("--ugc-append-mode", action="store_true",
                   help="Append adversarial text to first snippet seamlessly (default: off).")
    args = p.parse_args()

    _load_secrets()

    # Load UGC config if provided
    ugc_rules: dict[str, dict] = {}
    ugc_separator = "--- Additional comment excerpt ---"
    if args.ugc_config:
        ugc_data = json.loads(args.ugc_config.read_text(encoding="utf-8"))
        ugc_rules = ugc_data.get("rules_by_question_id", {})
        ugc_separator = ugc_data.get("separator", ugc_separator)
        print(f"UGC config: {args.ugc_config} ({len(ugc_rules)} rules)")

    with args.dataset.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    tasks = []
    for row in rows:
        geo_id = (row.get("geo_id") or "").strip()
        qid = (row.get("question_id") or "").strip()
        topic = (row.get("query") or "").strip()
        if not geo_id or not qid or not topic:
            continue
        # If UGC config provided, only run queries that have rules
        if ugc_rules and qid not in ugc_rules:
            continue
        tasks.append({
            "topic": topic,
            "run_id": f"{geo_id}__{qid}",
            "question_id": qid,
            "ugc_rule": ugc_rules.get(qid),
        })

    if args.limit > 0:
        tasks = tasks[: args.limit]

    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {len(tasks)} queries with {args.workers} workers -> {output_dir}")
    print(f"LLM={args.llm}, Retriever={args.retriever}, MindMap depth={args.depth}, retriever_k={args.retriever_k}, retrieve_top_k={args.retrieve_top_k}")

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                topic=t["topic"],
                run_id=t["run_id"],
                output_dir=output_dir,
                llm_model=args.llm,
                retriever_type=args.retriever,
                ugc_rule=t.get("ugc_rule"),
                ugc_separator=ugc_separator,
                ugc_append_mode=args.ugc_append_mode,
                enable_arctic_shift=args.enable_arctic_shift,
                merge_snippets=args.merge_snippets,
                depth=args.depth,
                retriever_k=args.retriever_k,
                retrieve_top_k=args.retrieve_top_k,
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
