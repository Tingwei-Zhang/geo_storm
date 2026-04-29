#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Domain sets stay intentionally simple and extensible.
SOCIAL_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "twitch.tv",
    "twitter.com",
    "x.com",
    "youtube.com",
    "discord.com",
}

COMMUNITY_DOMAINS = {
    "reddit.com",
    "old.reddit.com",
    "quora.com",
    "groups.google.com",
    "medium.com",
    "substack.com",
}

STACK_EXCHANGE_DOMAINS = {
    "stackexchange.com",
    "stackoverflow.com",
    "superuser.com",
    "serverfault.com",
    "mathoverflow.net",
    "askubuntu.com",
    "stackprinter.appspot.com",
}

REFERENCE_DOMAINS = {
    "wikipedia.org",
}

CODE_COMMUNITY_DOMAINS = {
    "github.com",
    "gitlab.com",
    "bitbucket.org",
}

UGC_BASES = (
    SOCIAL_DOMAINS
    | COMMUNITY_DOMAINS
    | STACK_EXCHANGE_DOMAINS
    | REFERENCE_DOMAINS
    | CODE_COMMUNITY_DOMAINS
)


def extract_cluster_name(question_id: str) -> str:
    """Convert grouped_best_mexican_food_0 -> best_mexican_food."""
    base, sep, suffix = question_id.rpartition("_")
    if sep and suffix.isdigit():
        question_id = base
    if question_id.startswith("grouped_"):
        return question_id[len("grouped_") :]
    return question_id


def normalize_url_base(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        return netloc[4:]
    return netloc


def canonicalize_url(url: str) -> str:
    """Canonicalize URL for stable overlap counting."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return ""

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]

    port = parsed.port
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if (port is None or default_port) else f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if not path.startswith("/"):
        path = "/" + path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Keep query semantics but make ordering stable for canonical equality.
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(query_pairs), doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def host_is_allowed(host: str, allowed_domains: set[str]) -> bool:
    """Match exact domain or any subdomain of an allowed base."""
    h = host.lower().strip(".")
    return any(h == d or h.endswith("." + d) for d in allowed_domains)


def is_ugc_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    return host_is_allowed(host, UGC_BASES)


def _urls_from_raw_retrieved_info(items: object) -> list[str]:
    urls: list[str] = []
    if not isinstance(items, list):
        return urls
    for info in items:
        if isinstance(info, dict):
            url = info.get("url")
            if isinstance(url, str) and url.strip():
                urls.append(url.strip())
    return urls


def iter_all_retrieved_urls(payload: object) -> list[str]:
    """Mirror notebook 'all retrieved' logic as closely as possible."""
    if not isinstance(payload, dict):
        return []

    urls: list[str] = []
    for item in payload.get("conversation_history") or []:
        if isinstance(item, dict):
            urls.extend(_urls_from_raw_retrieved_info(item.get("raw_retrieved_info")))
    for item in payload.get("warmstart_conv_archive") or []:
        if isinstance(item, dict):
            urls.extend(_urls_from_raw_retrieved_info(item.get("raw_retrieved_info")))

    for key in ("info_uuid_to_info_dict",):
        info_dict = payload.get(key)
        if isinstance(info_dict, dict):
            for val in info_dict.values():
                if isinstance(val, dict):
                    u = val.get("url")
                    if isinstance(u, str) and u.strip():
                        urls.append(u.strip())

    knowledge_base = payload.get("knowledge_base")
    if isinstance(knowledge_base, dict):
        info_dict = knowledge_base.get("info_uuid_to_info_dict")
        if isinstance(info_dict, dict):
            for val in info_dict.values():
                if isinstance(val, dict):
                    u = val.get("url")
                    if isinstance(u, str) and u.strip():
                        urls.append(u.strip())
    return urls


def iter_used_urls(payload: object, report_path: Path) -> list[str]:
    """Extract only URLs whose citation numbers appear in report.md."""
    knowledge_base = payload.get("knowledge_base", {}) if isinstance(payload, dict) else {}
    info_dict = (
        knowledge_base.get("info_uuid_to_info_dict", {})
        if isinstance(knowledge_base, dict)
        else {}
    )
    if not isinstance(info_dict, dict) or not report_path.exists():
        return []

    try:
        report_text = report_path.read_text(encoding="utf-8")
    except OSError:
        return []

    used_citation_ids = {int(x) for x in re.findall(r"\[(\d+)\]", report_text)}
    if not used_citation_ids:
        return []

    urls: list[str] = []
    for info in info_dict.values():
        if not isinstance(info, dict):
            continue
        citation_uuid = info.get("citation_uuid")
        url = info.get("url")
        if (
            isinstance(citation_uuid, int)
            and citation_uuid in used_citation_ids
            and isinstance(url, str)
            and url.strip()
        ):
            urls.append(url.strip())
    return urls


def generate_rows(
    run_root: Path, min_count: int, url_mode: str, domain_mode: str
) -> list[dict[str, object]]:
    by_cluster_url_questions: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    run_dirs = sorted(
        p
        for p in run_root.iterdir()
        if p.is_dir() and (p / "instance_dump.json").exists()
    )
    for run_dir in run_dirs:
        question_id = run_dir.name
        cluster_name = extract_cluster_name(question_id)
        instance_dump_path = run_dir / "instance_dump.json"
        if not instance_dump_path.exists():
            continue

        try:
            payload = json.loads(instance_dump_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if url_mode == "all":
            candidate_urls = iter_all_retrieved_urls(payload)
        else:
            candidate_urls = iter_used_urls(payload, run_dir / "report.md")

        # Count each URL at most once per question to measure overlap across questions.
        unique_urls_for_question: set[str] = set()
        for url in candidate_urls:
            normalized = canonicalize_url(url)
            if normalized:
                if domain_mode == "ugc_only" and not is_ugc_url(normalized):
                    continue
                unique_urls_for_question.add(normalized)
        for url in unique_urls_for_question:
            by_cluster_url_questions[cluster_name][url].add(question_id)

    rows: list[dict[str, object]] = []
    for cluster_name, url_map in by_cluster_url_questions.items():
        for url, qids in url_map.items():
            if len(qids) < min_count:
                continue
            sorted_qids = sorted(qids)
            rows.append(
                {
                    "cluster_name": cluster_name,
                    "question_ids": str(sorted_qids),
                    "url": url,
                    "url_base": normalize_url_base(url),
                    "count": len(sorted_qids),
                }
            )

    rows.sort(key=lambda r: (str(r["cluster_name"]), -int(r["count"]), str(r["url"])))
    return rows


def parse_question_ids(cell: str) -> set[str]:
    try:
        parsed = ast.literal_eval(cell)
    except (SyntaxError, ValueError):
        return set()
    if isinstance(parsed, list):
        return {str(v) for v in parsed}
    return set()


def compare_with_existing(existing_csv: Path, new_rows: list[dict[str, object]]) -> None:
    if not existing_csv.exists():
        return

    with existing_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing = {}
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            existing[url] = {
                "count": int((row.get("count") or "0").strip() or "0"),
                "question_ids": parse_question_ids(row.get("question_ids") or "[]"),
            }

    new_map = {
        str(r["url"]): {
            "count": int(r["count"]),
            "question_ids": parse_question_ids(str(r["question_ids"])),
        }
        for r in new_rows
    }

    overlap = set(existing) & set(new_map)
    same_count = sum(1 for u in overlap if existing[u]["count"] == new_map[u]["count"])
    same_qids = sum(
        1 for u in overlap if existing[u]["question_ids"] == new_map[u]["question_ids"]
    )
    print(f"Compare against {existing_csv}:")
    print(f"  existing rows (by url): {len(existing)}")
    print(f"  new rows (by url): {len(new_map)}")
    print(f"  shared urls: {len(overlap)}")
    print(f"  shared urls with same count: {same_count}")
    print(f"  shared urls with same question_ids set: {same_qids}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan run folders and build recurring URL overlap CSV by cluster name."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("geo_out/clean_run_hal_dataset"),
        help="Directory containing run folders (each with instance_dump.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("geo_out/recurring_urls_raw_grouped_hal_rebuilt.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Only keep URLs that appear in at least this many question runs.",
    )
    parser.add_argument(
        "--url-mode",
        choices=["all", "used"],
        default="all",
        help=(
            "all: include all retrieved URLs from raw_retrieved_info. "
            "used: include only URLs whose citation IDs appear in report.md."
        ),
    )
    parser.add_argument(
        "--domain-mode",
        choices=["all", "ugc_only"],
        default="all",
        help="all: keep all domains. ugc_only: keep only URLs in UGC_BASES.",
    )
    parser.add_argument(
        "--compare-existing",
        type=Path,
        default=Path("geo_out/recurring_urls_raw_grouped_hal.csv"),
        help="Optional reference CSV for quick overlap comparison.",
    )
    args = parser.parse_args()

    rows = generate_rows(args.run_root, args.min_count, args.url_mode, args.domain_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cluster_name", "question_ids", "url", "url_base", "count"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Mode: {args.url_mode}")
    print(f"Domains: {args.domain_mode}")
    print(f"Wrote {len(rows)} rows to {args.output}")
    compare_with_existing(args.compare_existing, rows)


if __name__ == "__main__":
    main()
