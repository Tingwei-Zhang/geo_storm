#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from examples.geo_examples.build_recurring_urls_from_runs import (
        canonicalize_url,
        extract_cluster_name,
        is_ugc_url,
        iter_all_retrieved_urls,
        iter_used_urls,
    )
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples.build_recurring_urls_from_runs import (  # type: ignore
        canonicalize_url,
        extract_cluster_name,
        is_ugc_url,
        iter_all_retrieved_urls,
        iter_used_urls,
    )


SEPARATOR = "--- Additional comment excerpt ---"


def _extract_adversarial_text(geo_document: str) -> str:
    try:
        obj = json.loads(geo_document)
    except json.JSONDecodeError:
        return ""
    if not isinstance(obj, dict):
        return ""
    content = obj.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    snippets = obj.get("snippets")
    if isinstance(snippets, list):
        parts = [s.strip() for s in snippets if isinstance(s, str) and s.strip()]
        if parts:
            return "\n\n".join(parts)
    description = obj.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    title = obj.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-query top UGC URL mapping where each query chooses among URLs "
            "that appeared in that query's own generation; score by cluster frequency."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("geo_out/clean_run_hal_dataset"),
        help="Run folders containing instance_dump.json and optional report.md.",
    )
    parser.add_argument(
        "--url-mode",
        choices=["all", "used"],
        default="all",
        help="all: use all retrieved URLs. used: only citation-used URLs from report.md.",
    )
    parser.add_argument(
        "--geo-dataset-csv",
        type=Path,
        default=Path("geo_out/geo_dataset_manifest_hal_clean_20260324_022028_6c91.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("geo_out/top_overlap_ugc_by_query.csv"),
        help="New output CSV (cleaner than mutating recurring overlap CSV).",
    )
    parser.add_argument(
        "--out-config-json",
        type=Path,
        default=Path("geo_out/ugc_mimic_config_top_overlap.json"),
    )
    args = parser.parse_args()

    if not args.run_root.exists():
        raise FileNotFoundError(f"Run root not found: {args.run_root}")
    if not args.geo_dataset_csv.exists():
        raise FileNotFoundError(f"GEO dataset CSV not found: {args.geo_dataset_csv}")

    # query_id -> set(ugc_urls observed in this query)
    query_urls: dict[str, set[str]] = {}
    # cluster -> url -> set(query_ids) to get per-cluster URL frequency
    cluster_url_queries: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    run_dirs = sorted(
        p for p in args.run_root.iterdir() if p.is_dir() and (p / "instance_dump.json").exists()
    )
    for run_dir in run_dirs:
        question_id = run_dir.name
        cluster = extract_cluster_name(question_id)
        payload = json.loads((run_dir / "instance_dump.json").read_text(encoding="utf-8"))
        if args.url_mode == "used":
            urls = iter_used_urls(payload, run_dir / "report.md")
        else:
            urls = iter_all_retrieved_urls(payload)

        url_set: set[str] = set()
        for raw_url in urls:
            normalized = canonicalize_url(raw_url)
            if normalized and is_ugc_url(normalized):
                url_set.add(normalized)
        query_urls[question_id] = url_set
        for u in url_set:
            cluster_url_queries[cluster][u].add(question_id)

    out_rows: list[dict[str, str]] = []
    rules_by_qid: dict[str, dict[str, str]] = {}

    with args.geo_dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("question_id") or "").strip()
            cluster_id = (row.get("cluster_id") or "").strip()
            geo_id = (row.get("geo_id") or "").strip()
            query = (row.get("query") or "").strip()
            adversarial_text = _extract_adversarial_text((row.get("geo_document") or "").strip())
            cluster = cluster_id or extract_cluster_name(qid)

            top_url = "none"
            top_count = "0"
            candidates = query_urls.get(qid, set())
            if candidates:
                scored = []
                for u in candidates:
                    c = len(cluster_url_queries.get(cluster, {}).get(u, set()))
                    scored.append((c, u))
                scored.sort(key=lambda x: (-x[0], x[1]))
                best_count, best_url = scored[0]
                top_url = best_url
                top_count = str(best_count)

            out_rows.append(
                {
                    "question_id": qid,
                    "cluster_id": cluster_id,
                    "geo_id": geo_id,
                    "query": query,
                    "top_overlap_ugc_url": top_url,
                    "top_overlap_count": top_count,
                    "adversarial_text": adversarial_text,
                }
            )

            if top_url != "none" and adversarial_text:
                rules_by_qid[qid] = {
                    "target_url": top_url,
                    "adversarial_text": adversarial_text,
                }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "cluster_id",
                "geo_id",
                "query",
                "top_overlap_ugc_url",
                "top_overlap_count",
                "adversarial_text",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    config = {
        "separator": SEPARATOR,
        "rules_by_question_id": rules_by_qid,
        "selection_policy": (
            "For each query, pick the URL with highest cluster frequency among UGC URLs "
            "that appeared in that query generation. Allows count=1 fallback."
        ),
        "url_mode": args.url_mode,
    }
    args.out_config_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_config_json.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    none_count = sum(1 for r in out_rows if r["top_overlap_ugc_url"] == "none")
    print(f"Wrote: {args.out_csv} ({len(out_rows)} rows; none={none_count})")
    print(f"Wrote: {args.out_config_json} ({len(rules_by_qid)} question rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
