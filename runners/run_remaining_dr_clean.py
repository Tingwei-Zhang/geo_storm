#!/usr/bin/env python3
"""
Run remaining OpenAI DR and Gemini DR clean runs.

Checks existing runs for completeness (article must be >= 200 chars),
marks short/refused articles as incomplete (renames to article.txt.refused
so the runner sees no article.txt and re-runs them), then launches the
batch runners for missing queries.

Usage:
    cd geo_storm

    # Dry run — just print what would happen:
    python run_remaining_dr_clean.py --dry-run

    # Actually run OpenAI DR remaining queries:
    python run_remaining_dr_clean.py --run openai

    # Actually run Gemini DR remaining queries:
    python run_remaining_dr_clean.py --run gemini

    # Run both:
    python run_remaining_dr_clean.py --run both
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

MIN_ARTICLE_LEN = 200  # chars — below this is considered refused/empty

DATASET = Path("geo_out/geo_dataset_clean.csv")
OPENAI_DIR = Path("geo_out/openai_dr/clean_runs")
GEMINI_DIR = Path("geo_out/gemini_dr/clean_runs")


def load_dataset() -> list[dict]:
    with DATASET.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def audit_runs(output_dir: Path, dataset: list[dict]) -> dict:
    """Audit existing runs. Returns dict with complete/incomplete/missing lists."""
    complete = []
    incomplete = []  # exists but article too short or missing
    missing = []

    for row in dataset:
        geo_id = row["geo_id"].strip()
        qid = row["question_id"].strip()
        cluster = row["cluster_id"].strip()
        run_id = f"{geo_id}__{qid}"
        run_dir = output_dir / run_id

        if not run_dir.exists():
            missing.append({"run_id": run_id, "cluster": cluster, "reason": "no directory"})
            continue

        article = run_dir / "article.txt"
        if not article.exists():
            # Check for previously-renamed refused articles
            refused = run_dir / "article.txt.refused"
            if refused.exists():
                incomplete.append({"run_id": run_id, "cluster": cluster, "reason": "refused (already marked)"})
            else:
                incomplete.append({"run_id": run_id, "cluster": cluster, "reason": "no article.txt"})
            continue

        text = article.read_text(encoding="utf-8").strip()
        if len(text) < MIN_ARTICLE_LEN:
            incomplete.append({
                "run_id": run_id,
                "cluster": cluster,
                "reason": f"article too short ({len(text)} chars): {text[:80]}...",
            })
            continue

        complete.append({"run_id": run_id, "cluster": cluster})

    return {"complete": complete, "incomplete": incomplete, "missing": missing}


def mark_refused(output_dir: Path, incomplete: list[dict], dry_run: bool):
    """Rename short/refused articles so the runner will re-attempt them.

    SAFETY: Only renames truly empty articles (0 chars).  Short articles
    that contain refusal text (e.g., "I'm sorry...") are left as-is so
    skip_if_exists will skip them—re-running would just produce the same
    refusal and waste money.
    """
    marked = 0
    for item in incomplete:
        run_dir = output_dir / item["run_id"]
        article = run_dir / "article.txt"
        refused = run_dir / "article.txt.refused"
        if article.exists() and not refused.exists():
            text = article.read_text(encoding="utf-8").strip()
            if len(text) > 0:
                # Has content (likely a refusal) — leave it so skip_if_exists works
                continue
            if dry_run:
                print(f"  [DRY RUN] Would rename {article} -> {refused}")
            else:
                article.rename(refused)
                print(f"  Renamed {article} -> article.txt.refused")
            marked += 1
    return marked


def print_audit(name: str, audit: dict):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Complete:   {len(audit['complete'])}")
    print(f"  Incomplete: {len(audit['incomplete'])}")
    print(f"  Missing:    {len(audit['missing'])}")
    print(f"  To run:     {len(audit['incomplete']) + len(audit['missing'])}")

    # Per-cluster summary
    clusters = {}
    for item in audit["complete"]:
        clusters.setdefault(item["cluster"], {"c": 0, "i": 0, "m": 0})
        clusters[item["cluster"]]["c"] += 1
    for item in audit["incomplete"]:
        clusters.setdefault(item["cluster"], {"c": 0, "i": 0, "m": 0})
        clusters[item["cluster"]]["i"] += 1
    for item in audit["missing"]:
        clusters.setdefault(item["cluster"], {"c": 0, "i": 0, "m": 0})
        clusters[item["cluster"]]["m"] += 1

    print(f"\n  {'Cluster':<40s} {'Done':>5} {'Inc':>5} {'Miss':>5}")
    for cl in sorted(clusters):
        s = clusters[cl]
        total = s["c"] + s["i"] + s["m"]
        print(f"  {cl:<40s} {s['c']:>5} {s['i']:>5} {s['m']:>5}  (/{total})")

    if audit["incomplete"]:
        print(f"\n  Incomplete details:")
        for item in audit["incomplete"]:
            print(f"    {item['run_id']}: {item['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Run remaining DR clean runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print what would be done, don't run anything.")
    parser.add_argument("--run", choices=["openai", "gemini", "both", "none"],
                        default="none",
                        help="Which system(s) to run. 'none' = audit only.")
    parser.add_argument("--no-mark-refused", action="store_true",
                        help="Don't rename refused/short articles.")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} queries")

    # Audit
    oai_audit = audit_runs(OPENAI_DIR, dataset)
    gem_audit = audit_runs(GEMINI_DIR, dataset)

    print_audit("OpenAI Deep Research", oai_audit)
    print_audit("Gemini Deep Research", gem_audit)

    if args.run == "none" and not args.dry_run:
        print("\nUse --run openai|gemini|both to actually run queries.")
        return

    # Mark refused articles
    if not args.no_mark_refused:
        if args.run in ("openai", "both") or args.dry_run:
            n = mark_refused(OPENAI_DIR, oai_audit["incomplete"], args.dry_run)
            if n:
                print(f"\nMarked {n} OpenAI DR refused articles for re-run.")

        if args.run in ("gemini", "both") or args.dry_run:
            n = mark_refused(GEMINI_DIR, gem_audit["incomplete"], args.dry_run)
            if n:
                print(f"\nMarked {n} Gemini DR refused articles for re-run.")

    if args.dry_run:
        print("\n[DRY RUN] Would run the batch scripts above. Use --run to execute.")
        return

    # Run OpenAI DR
    if args.run in ("openai", "both"):
        to_run = len(oai_audit["incomplete"]) + len(oai_audit["missing"])
        if to_run == 0:
            print("\nOpenAI DR: nothing to run.")
        else:
            print(f"\nRunning OpenAI DR ({to_run} queries)...")
            cmd = [
                sys.executable, "-m", "examples.openai_dr_batch.run_openai_dr_batch",
                "--dataset", str(DATASET),
                "--output-dir", str(OPENAI_DIR),
            ]
            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd)

    # Run Gemini DR
    if args.run in ("gemini", "both"):
        to_run = len(gem_audit["incomplete"]) + len(gem_audit["missing"])
        if to_run == 0:
            print("\nGemini DR: nothing to run.")
        else:
            print(f"\nRunning Gemini DR ({to_run} queries)...")
            cmd = [
                sys.executable, "-m", "examples.gemini_dr_batch.run_gemini_dr_batch",
                "--dataset", str(DATASET),
                "--output-dir", str(GEMINI_DIR),
            ]
            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd)


if __name__ == "__main__":
    main()
