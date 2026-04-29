#!/usr/bin/env python3
"""
UGC-aware evaluation for GEO runs.

Computes:
  - `cited` / `mentioned` exactly like `examples/geo_examples/evaluate_geo.py`
  - `ugc_exposure` from `ugc_mimic_meta.json` produced by the UGC-mimic runner
  - conditional success rates: cited/mentioned given ugc_exposure=true
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    # When executed from repo root with PYTHONPATH configured, this works.
    from examples.geo_examples import evaluate_geo as eval_geo  # type: ignore
except ModuleNotFoundError:
    # Fallback for direct script execution.
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples import evaluate_geo as eval_geo  # type: ignore


def _domain_from_row(row: dict[str, Any], question_id: str) -> str:
    return (
        (row.get("cluster_id") or row.get("group_id") or "").strip()
        or (question_id.split("_", maxsplit=1)[0] if question_id else "unknown")
    )


def _target_name_for_domain(domain: str) -> str:
    tn = getattr(eval_geo, "TARGET_NAMES", {})
    if isinstance(tn, dict):
        return str(tn.get(domain, "")).strip()
    return ""


def _ugc_exposure(run_dir: Path) -> bool:
    meta_path = run_dir / "ugc_mimic_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("ugc_exposure"):
            return True
        if meta.get("patched_count", 0) > 0:
            return True
        if meta.get("matched_urls"):
            return True
        return False
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate GEO runs with ugc_exposure-aware metrics.")
    p.add_argument("dataset_csv", type=Path, help="Manifest CSV used to map into run directories.")
    p.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory.")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    p.add_argument(
        "--denominator",
        choices=["existing", "all"],
        default="existing",
        help="Metric denominator for summary rates (default: existing run dirs only).",
    )
    args = p.parse_args()

    dataset_csv: Path = args.dataset_csv
    base_no_suffix = dataset_csv.with_suffix("")
    runs_dir = args.runs_dir or Path(f"{base_no_suffix}_runs")
    output_csv = args.output or Path(f"{base_no_suffix}_ugc_evaluated.csv")

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Input dataset CSV not found: {dataset_csv}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)
        input_fieldnames = list(reader.fieldnames or [])

    out_fieldnames = [
        *input_fieldnames,
        "ugc_exposure",
        "cited",
        "mentioned",
        "cited_if_exposed",
        "mentioned_if_exposed",
    ]

    # Overall counters
    total = len(input_rows)
    existing = 0
    exposed = 0
    cited = 0
    mentioned = 0
    cited_exposed = 0
    mentioned_exposed = 0

    by_domain_total: dict[str, int] = defaultdict(int)
    by_domain_exposed: dict[str, int] = defaultdict(int)
    by_domain_cited: dict[str, int] = defaultdict(int)
    by_domain_mentioned: dict[str, int] = defaultdict(int)
    by_domain_cited_exposed: dict[str, int] = defaultdict(int)
    by_domain_mentioned_exposed: dict[str, int] = defaultdict(int)

    output_rows: list[dict[str, str]] = []

    for row in input_rows:
        geo_id = (row.get("geo_id") or "").strip()
        question_id = (row.get("question_id") or "").strip()
        domain = _domain_from_row(row, question_id)
        target_name = _target_name_for_domain(domain)

        run_dir = runs_dir / f"{geo_id}__{question_id}"
        run_exists = run_dir.exists()
        if run_exists:
            existing += 1
        exp = _ugc_exposure(run_dir)

        # Mirror evaluate_geo.py logic.
        if run_exists and target_name:
            cited_run, mentioned_run = eval_geo.compute_metrics_for_run(run_dir, target_name)
        else:
            cited_run, mentioned_run = False, False

        if run_exists:
            by_domain_total[domain] += 1
        if exp:
            by_domain_exposed[domain] += 1

        if cited_run:
            cited += 1
            by_domain_cited[domain] += 1
            if exp:
                cited_exposed += 1
                by_domain_cited_exposed[domain] += 1
        if mentioned_run:
            mentioned += 1
            by_domain_mentioned[domain] += 1
            if exp:
                mentioned_exposed += 1
                by_domain_mentioned_exposed[domain] += 1
        if exp:
            exposed += 1

        output_row = dict(row)
        output_row["ugc_exposure"] = "true" if exp else "false"
        output_row["cited"] = "true" if cited_run else "false"
        output_row["mentioned"] = "true" if mentioned_run else "false"
        output_row["cited_if_exposed"] = "true" if (exp and cited_run) else "false"
        output_row["mentioned_if_exposed"] = "true" if (exp and mentioned_run) else "false"
        output_rows.append(output_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    def _rate(n: int, d: int) -> str:
        return f"{n}/{d} ({(100.0 * n / d):.1f}%)" if d else "0/0 (0.0%)"

    print(f"Dataset: {dataset_csv}")
    print(f"Runs: {runs_dir}")
    print(f"Wrote: {output_csv}")
    print(f"Run coverage: {existing}/{total} ({(100.0 * existing / total):.1f}%)")
    print()
    denom = existing if args.denominator == "existing" else total
    denom_label = "existing runs" if args.denominator == "existing" else "all dataset rows"
    print("Overall:")
    print(f"  denominator: {denom_label}")
    print(f"  ugc_exposure=true: {_rate(exposed, denom)}")
    print(f"  cited: {_rate(cited, denom)}")
    print(f"  mentioned: {_rate(mentioned, denom)}")
    if exposed:
        print("  conditional on ugc_exposure=true:")
        print(f"    cited: {_rate(cited_exposed, exposed)}")
        print(f"    mentioned: {_rate(mentioned_exposed, exposed)}")
    else:
        print("  conditional on ugc_exposure=true: n/a (no exposed runs)")

    print("\nBy cluster:")
    for domain in sorted(by_domain_total):
        tot = by_domain_total[domain]
        exp_d = by_domain_exposed[domain]
        print(
            f"  {domain}: exposed={_rate(exp_d, tot)}, "
            f"cited={_rate(by_domain_cited[domain], tot)}, "
            f"mentioned={_rate(by_domain_mentioned[domain], tot)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

