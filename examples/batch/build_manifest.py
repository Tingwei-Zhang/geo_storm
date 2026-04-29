"""
Build a batch manifest from seo-geo cluster CSVs.

Reads *_clusters.csv from a local directory or S3 prefix, adds a `topic` column
(configurable from title and/or body), optionally joins injection doc paths/URIs
from a mapping file, and writes manifest CSV (and optionally JSONL).
Duplicate question_id: first occurrence wins.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator

# Optional S3 support
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


# Expected CSV columns from seo-geo cluster files (normalize to lowercase for matching)
CLUSTER_COLUMNS = [
    "question_id",
    "title",
    "body",
    "questionviewcount",
    "questionscore",
    "questioncreationdate",
    "domain",
    "cluster_id",
    "cluster_probability",
    "cluster_size",
]


def _normalize_row(raw: dict) -> dict:
    """Normalize keys to lowercase and strip whitespace from string values."""
    out = {}
    for k, v in raw.items():
        key = k.strip().lower() if isinstance(k, str) else k
        if isinstance(v, str):
            out[key] = v.strip()
        else:
            out[key] = v
    return out


def _topic_from_row(
    row: dict,
    topic_from: str,
    body_max_chars: int,
) -> str:
    """Build topic string from row. topic_from: 'title' | 'title_and_body'."""
    title = (row.get("title") or "").strip()
    body = (row.get("body") or "").strip()

    if topic_from == "title":
        return title or "unknown"
    if topic_from == "title_and_body":
        if not body:
            return title or "unknown"
        body_trunc = body[:body_max_chars] + ("..." if len(body) > body_max_chars else "")
        combined = f"{title} {body_trunc}".strip()
        return combined or "unknown"
    return title or "unknown"


def _dataset_name_from_path(path: Path) -> str:
    """Derive dataset name from CSV path, e.g. health_clusters.csv -> health."""
    stem = path.stem.lower()
    if stem.endswith("_clusters"):
        return stem[: -len("_clusters")]
    return stem


def _dataset_name_from_s3_uri(s3_uri: str) -> str:
    """Derive dataset name from S3 key, e.g. .../health_clusters.csv -> health."""
    key = s3_uri.split("/")[-1] if "/" in s3_uri else s3_uri
    stem = key.replace(".csv", "").lower()
    if stem.endswith("_clusters"):
        return stem[: -len("_clusters")]
    return stem


def _list_local_csvs(input_dir: Path) -> list[Path]:
    """List *_clusters.csv under input_dir."""
    if not input_dir.is_dir():
        return []
    return sorted(input_dir.glob("*_clusters.csv"))


def _read_local_csv(path: Path) -> Iterator[dict]:
    """Yield normalized rows from a local CSV."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _normalize_row(row)


def _read_csv_from_s3(s3_uri: str) -> Iterator[dict]:
    """Yield normalized rows from a CSV stored in S3. s3_uri e.g. s3://bucket/key."""
    if not HAS_BOTO3:
        raise RuntimeError("Reading from S3 requires boto3. Install with: pip install boto3")
    match = re.match(r"s3://([^/]+)/(.+)$", s3_uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, key = match.group(1), match.group(2)
    client = boto3.client("s3")
    resp = client.get_object(Bucket=bucket, Key=key)
    text = resp["Body"].read().decode("utf-8", errors="replace")
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        yield _normalize_row(row)


def _list_s3_csvs(prefix_uri: str) -> list[str]:
    """List s3 URIs for *_clusters.csv under prefix. prefix_uri e.g. s3://bucket/prefix/."""
    if not HAS_BOTO3:
        raise RuntimeError("S3 listing requires boto3. Install with: pip install boto3")
    match = re.match(r"s3://([^/]+)/(.*)$", prefix_uri.rstrip("/"))
    if not match:
        raise ValueError(f"Invalid S3 prefix: {prefix_uri}")
    bucket, prefix = match.group(1), match.group(2)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    uris = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if "_clusters.csv" in key and key.endswith(".csv"):
                uris.append(f"s3://{bucket}/{key}")
    return sorted(uris)


def _load_injection_mapping(mapping_path: Path) -> dict[str, dict]:
    """Load injection mapping CSV: question_id -> {injection_doc_path?, injection_doc_s3_uri?}.
    Returns dict keyed by question_id (first occurrence wins).
    """
    if not mapping_path.exists():
        return {}
    out = {}
    with open(mapping_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = _normalize_row(row)
            qid = (row.get("question_id") or "").strip()
            if not qid or qid in out:
                continue
            out[qid] = {
                "injection_doc_path": (row.get("injection_doc_path") or "").strip() or None,
                "injection_doc_s3_uri": (row.get("injection_doc_s3_uri") or "").strip() or None,
            }
    return out


def build_manifest(
    input_dir: Path | None = None,
    input_s3_prefix: str | None = None,
    topic_from: str = "title",
    body_max_chars: int = 500,
    injection_mapping_path: Path | None = None,
    duplicate_policy: str = "first_wins",
) -> list[dict]:
    """
    Build manifest rows from cluster CSVs.
    question_id is set to dataset_name + "_" + original question_id (e.g. health_123)
    so ids are unique across datasets. original_question_id is kept for reference.
    duplicate_policy: 'first_wins' (keep first row per question_id) or 'fail_fast'.
    """
    seen_qids = set()
    manifest_rows = []
    injection_map = _load_injection_mapping(injection_mapping_path) if injection_mapping_path else {}

    def process_row(row: dict, dataset_name: str) -> None:
        raw_qid = (row.get("question_id") or "").strip()
        if not raw_qid:
            return
        qid = f"{dataset_name}_{raw_qid}"
        if duplicate_policy == "fail_fast" and qid in seen_qids:
            raise ValueError(f"Duplicate question_id in manifest: {qid}")
        if qid in seen_qids:
            return
        seen_qids.add(qid)
        topic = _topic_from_row(row, topic_from, body_max_chars)
        out = {
            "question_id": qid,
            "original_question_id": raw_qid,
            "dataset": dataset_name,
            "topic": topic,
            "title": row.get("title") or "",
            "body": row.get("body") or "",
            "domain": row.get("domain") or "",
            "cluster_id": row.get("cluster_id") or "",
        }
        if qid in injection_map:
            m = injection_map[qid]
            out["injection_doc_path"] = m.get("injection_doc_path") or ""
            out["injection_doc_s3_uri"] = m.get("injection_doc_s3_uri") or ""
        else:
            out["injection_doc_path"] = ""
            out["injection_doc_s3_uri"] = ""
        manifest_rows.append(out)

    if input_dir is not None:
        for csv_path in _list_local_csvs(input_dir):
            dataset_name = _dataset_name_from_path(csv_path)
            for row in _read_local_csv(csv_path):
                process_row(row, dataset_name)
    elif input_s3_prefix is not None:
        for s3_uri in _list_s3_csvs(input_s3_prefix):
            dataset_name = _dataset_name_from_s3_uri(s3_uri)
            for row in _read_csv_from_s3(s3_uri):
                process_row(row, dataset_name)
    else:
        raise ValueError("Provide either input_dir or input_s3_prefix")

    return manifest_rows


def write_manifest_csv(rows: list[dict], path: Path) -> None:
    """Write manifest to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        list(rows[0].keys())
        if rows
        else [
            "question_id",
            "original_question_id",
            "dataset",
            "topic",
            "title",
            "body",
            "domain",
            "cluster_id",
            "injection_doc_path",
            "injection_doc_s3_uri",
        ]
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_jsonl(rows: list[dict], path: Path) -> None:
    """Write manifest to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    """Upload a file to S3."""
    if not HAS_BOTO3:
        raise RuntimeError("S3 upload requires boto3. Install with: pip install boto3")
    match = re.match(r"s3://([^/]+)/(.+)$", s3_uri.strip())
    if not match:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket, key = match.group(1), match.group(2)
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def main() -> None:
    parser = ArgumentParser(
        description="Build batch manifest from seo-geo cluster CSVs (local or S3)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Local directory containing *_clusters.csv files.",
    )
    parser.add_argument(
        "--input-s3",
        type=str,
        default=None,
        help="S3 prefix for cluster CSVs (e.g. s3://bucket/clustering_results/clusters/).",
    )
    parser.add_argument(
        "--topic-from",
        type=str,
        choices=["title", "title_and_body"],
        default="title",
        help="How to build topic: title only, or title + body truncated.",
    )
    parser.add_argument(
        "--body-max-chars",
        type=int,
        default=500,
        help="Max body chars when topic_from=title_and_body.",
    )
    parser.add_argument(
        "--injection-mapping",
        type=Path,
        default=None,
        help="Optional CSV with question_id, injection_doc_path, injection_doc_s3_uri.",
    )
    parser.add_argument(
        "--duplicate",
        type=str,
        choices=["first_wins", "fail_fast"],
        default="first_wins",
        help="Duplicate question_id: keep first or abort.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optionally write manifest as JSONL.",
    )
    parser.add_argument(
        "--upload-s3",
        type=str,
        default=None,
        help="After writing, upload manifest to this S3 URI (e.g. s3://bucket/batch/manifest.csv).",
    )
    args = parser.parse_args()

    if (args.input_dir is None) == (args.input_s3 is None):
        print("Provide exactly one of --input-dir or --input-s3.", file=sys.stderr)
        sys.exit(1)

    rows = build_manifest(
        input_dir=args.input_dir,
        input_s3_prefix=args.input_s3,
        topic_from=args.topic_from,
        body_max_chars=args.body_max_chars,
        injection_mapping_path=args.injection_mapping,
        duplicate_policy=args.duplicate,
    )
    write_manifest_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.output_jsonl is not None:
        write_manifest_jsonl(rows, args.output_jsonl)
        print(f"Wrote JSONL to {args.output_jsonl}")
    if args.upload_s3 is not None:
        upload_to_s3(args.output, args.upload_s3)
        print(f"Uploaded to {args.upload_s3}")


if __name__ == "__main__":
    main()
