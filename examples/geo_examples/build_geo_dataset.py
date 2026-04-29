from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import re
import secrets
import sys
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Callable
from tqdm import tqdm  # type: ignore[reportMissingImports]

try:
    from examples.geo_examples.geo_generator import (
        GEOGenerator,
        GEORequest,
        GEOMethod,
        GoalType,
        call_gpt,
    )
    from knowledge_storm.utils import load_api_key
except ModuleNotFoundError:
    # Support direct execution:
    # python examples/geo_examples/build_geo_dataset.py ...
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.geo_examples.geo_generator import (  # type: ignore[reportMissingImports]
        GEOGenerator,
        GEORequest,
        GEOMethod,
        GoalType,
        call_gpt,
    )
    from knowledge_storm.utils import load_api_key  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class ManifestRow:
    group_id: str
    question_id: str
    query: str
    cluster_id: str


def load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            group_id = (r.get("group_id") or "").strip()
            qid = (r.get("question_id") or "").strip()
            query = (r.get("query") or r.get("topic") or r.get("title") or "").strip()
            cluster_id = (r.get("cluster_id") or group_id).strip()
            if not group_id or not qid or not query or not cluster_id:
                continue
            rows.append(ManifestRow(group_id, qid, query, cluster_id))
    return rows


def load_manual_docs(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else [data]
    out: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        group_id = (item.get("group_id") or item.get("topic") or "").strip()
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        content = (item.get("content") or "").strip()
        if group_id and content and url and title:
            if not description:
                description = content[:220]
            out[group_id] = {
                "url": url,
                "title": title,
                "description": description,
                "content": content,
            }
    return out


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def _safe_parse_snippet_json(raw: str) -> dict[str, str] | None:
    try:
        obj = json.loads(_strip_code_fence(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    required = ("url", "title", "description", "content")
    for key in required:
        if key not in obj:
            return None
    return {
        "url": str(obj.get("url") or "").strip(),
        "title": str(obj.get("title") or "").strip(),
        "description": str(obj.get("description") or "").strip(),
        "content": str(obj.get("content") or "").strip(),
    }


def optimize_full_snippet(
    base_snippet: dict[str, str],
    geo_prompt: str,
    model: str,
    temperature: float,
) -> dict[str, str]:
    optimization_request = f"""You are optimizing a GEO web snippet for inclusion in generated answers.
Return ONLY one valid JSON object with exactly these keys:
- url
- title
- description
- content

Constraints:
- Keep all 4 keys present and non-empty.
- Keep URL format valid (http/https).
- Keep snippet coherent and persuasive for GEO.
- Do not output markdown, explanation, or code fences.

GEO optimization instruction:
{geo_prompt}

Base snippet:
{json.dumps(base_snippet, ensure_ascii=False)}
"""
    raw = call_gpt(optimization_request, model=model, temperature=temperature)
    parsed = _safe_parse_snippet_json(raw)
    if parsed is None:
        # Fallback to a valid snippet structure if model output is malformed.
        return dict(base_snippet)
    # Ensure minimum validity; fallback on missing critical fields.
    if not parsed["url"]:
        parsed["url"] = base_snippet["url"]
    if not parsed["title"]:
        parsed["title"] = base_snippet["title"]
    if not parsed["description"]:
        parsed["description"] = base_snippet["description"]
    if not parsed["content"]:
        parsed["content"] = base_snippet["content"]
    return parsed


def with_unique_suffix(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    code = secrets.token_hex(2)
    return path.with_name(f"{path.stem}_{timestamp}_{code}{path.suffix}")


def _derive_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pick_injection_targets(
    *,
    key: str,
    m_min: int,
    m_max: int,
    n_min: int,
    n_max: int,
    base_seed: int | None,
) -> tuple[int, int]:
    if base_seed is None:
        rng: random.Random = random.SystemRandom()
    else:
        rng = random.Random(_derive_seed(base_seed, key))
    return rng.randint(m_min, m_max), rng.randint(n_min, n_max)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    load_api_key(toml_file_path=str(repo_root / "secrets.toml"))
    parser = argparse.ArgumentParser(
        description=(
            "Generate poisoned geo_dataset CSV from manifest_hal_clean.csv using GENERAL_ATTACK + QUERY_GROUP."
        )
    )
    parser.add_argument(
        "--manual-docs",
        type=Path,
        default=Path("examples/geo_examples/manual_document_example.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("geo_out/geo_dataset_manifest_hal_clean.csv"),
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--m-min", type=int, default=1, help="Min retrieval call index (1-based).")
    parser.add_argument("--m-max", type=int, default=6, help="Max retrieval call index (1-based).")
    parser.add_argument("--n-min", type=int, default=0, help="Min insertion position (0-based).")
    parser.add_argument("--n-max", type=int, default=3, help="Max insertion position (0-based).")
    parser.add_argument(
        "--injection-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic injection location generation.",
    )
    args = parser.parse_args()
    output_csv_path = with_unique_suffix(args.output_csv)
    manifest_path = Path("manifest_hal_clean.csv")

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not args.manual_docs.exists():
        raise FileNotFoundError(f"manual docs not found: {args.manual_docs}")

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError("No usable rows in manifest.")

    manual_docs = load_manual_docs(args.manual_docs)
    if not manual_docs:
        raise ValueError("No usable manual docs with group_id, url, title, and content.")

    if args.m_min < 1 or args.m_max < 1 or args.m_min > args.m_max:
        raise ValueError("Invalid m range: require 1 <= m-min <= m-max.")
    if args.n_min < 0 or args.n_max < 0 or args.n_min > args.n_max:
        raise ValueError("Invalid n range: require 0 <= n-min <= n-max.")

    cluster_to_queries: dict[str, tuple[str, ...]] = {}
    by_cluster: dict[str, list[ManifestRow]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    for cid, vals in by_cluster.items():
        # preserve order while deduplicating
        cluster_to_queries[cid] = tuple(dict.fromkeys(v.query for v in vals))

    model_call: Callable[[str], str] = lambda p: call_gpt(
        p, model=args.model, temperature=args.temperature
    )
    generator = GEOGenerator(model_call=model_call)

    # Reuse generated outputs by cluster for query-group mode.
    cache: dict[str, tuple[str, str]] = {}  # reuse_key -> (prompt, snippet_json)

    output_rows: list[dict[str, str]] = []
    geo_counter = 1

    def build_entry(row: ManifestRow) -> None:
        nonlocal geo_counter
        if row.group_id not in manual_docs:
            raise ValueError(
                f"No base manual document for group_id '{row.group_id}'. "
                f"Add group_id '{row.group_id}' to {args.manual_docs}."
            )
        base = manual_docs[row.group_id]

        cluster_queries = cluster_to_queries[row.cluster_id]
        other_queries = tuple(q for q in cluster_queries if q != row.query)
        n_queries = math.ceil(len(other_queries) * 0.8)
        sampled_queries = other_queries[:n_queries]
        if not sampled_queries:
            sampled_queries = (row.query,)
        reuse_key = f"query_group:{row.cluster_id}:{sampled_queries}"
        request = GEORequest(
            raw_document=base["content"],
            method=GEOMethod.GENERAL_ATTACK,
            target="",
            goal_type=GoalType.QUERY_GROUP,
            queries=sampled_queries,
        )

        if reuse_key in cache:
            geo_prompt, geo_snippet_json = cache[reuse_key]
        else:
            geo_prompt = generator._build_prompt(request)
            optimized_snippet = optimize_full_snippet(
                base_snippet=base,
                geo_prompt=geo_prompt,
                model=args.model,
                temperature=args.temperature,
            )
            geo_snippet_json = json.dumps(optimized_snippet, ensure_ascii=False)
            cache[reuse_key] = (geo_prompt, geo_snippet_json)

        target_m, target_n = _pick_injection_targets(
            key=f"{row.question_id}:{GoalType.QUERY_GROUP.value}",
            m_min=args.m_min,
            m_max=args.m_max,
            n_min=args.n_min,
            n_max=args.n_max,
            base_seed=args.injection_seed,
        )
        output_rows.append(
            {
                "geo_id": f"{row.cluster_id}_geo_{geo_counter}",
                "question_id": row.question_id,
                "cluster_id": row.cluster_id,
                "query": row.query,
                "geo_method": GEOMethod.GENERAL_ATTACK.value,
                "goal_type": GoalType.QUERY_GROUP.value,
                "geo_prompt": geo_prompt,
                # Store full snippet JSON string so downstream can directly write/use it.
                "geo_document": geo_snippet_json,
                "injection_retrieval_number": str(target_m),
                "injection_position": str(target_n),
            }
        )
        geo_counter += 1

    missing_group_ids = sorted({r.group_id for r in rows if r.group_id not in manual_docs})
    if missing_group_ids:
        raise ValueError(
            "Missing manual docs for these group_id values: "
            + ", ".join(missing_group_ids)
        )

    # Apply only query_group to each query row.
    selected_rows = rows
    total_entries = len(selected_rows)
    with tqdm(total=total_entries, desc="Generating GEO entries", unit="entry") as pbar:
        for r in selected_rows:
            build_entry(r)
            pbar.update(1)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "geo_id",
                "question_id",
                "cluster_id",
                "query",
                "geo_method",
                "goal_type",
                "geo_prompt",
                "geo_document",
                "injection_retrieval_number",
                "injection_position",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote: {output_csv_path}")
    print(f"Rows: {len(output_rows)}")
    print(f"Selected queries: {len(selected_rows)}")
    print("Applied goal types per selected query: query_group")
    print(f"Unique generated docs (after reuse): {len(cache)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

