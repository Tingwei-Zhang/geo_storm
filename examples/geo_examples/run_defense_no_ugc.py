"""
Run Co-STORM with UGC domains blocked at retrieval time (defense experiment).

Blocks results from reddit.com, facebook.com, youtube.com, medium.com,
instagram.com, tiktok.com before they reach Co-STORM's article generator.
"""
from __future__ import annotations

import csv
import sys
from argparse import ArgumentParser
from pathlib import Path

try:
    from examples.batch.run_single_query import run_single_query
    from examples.batch._injector import UGC_DOMAINS
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.batch.run_single_query import run_single_query
    from examples.batch._injector import UGC_DOMAINS


def main() -> int:
    parser = ArgumentParser(description="Co-STORM defense: block UGC domains during retrieval.")
    parser.add_argument("--geo-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Max queries (0=all).")
    parser.add_argument("--clusters", nargs="*", default=None)
    parser.add_argument("--retriever", type=str, default="serper")
    parser.add_argument("--demo-turns", type=int, default=2)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--total-conv-turn", type=int, default=20)
    parser.add_argument("--max-search-queries", type=int, default=2)
    parser.add_argument("--max-search-thread", type=int, default=5)
    parser.add_argument("--max-search-queries-per-turn", type=int, default=3)
    parser.add_argument("--warmstart-max-num-experts", type=int, default=1)
    parser.add_argument("--warmstart-max-turn-per-experts", type=int, default=1)
    parser.add_argument("--warmstart-max-thread", type=int, default=1)
    parser.add_argument("--max-thread-num", type=int, default=5)
    parser.add_argument("--max-num-round-table-experts", type=int, default=1)
    parser.add_argument("--moderator-override-n", type=int, default=2)
    parser.add_argument("--node-expansion-trigger-count", type=int, default=10)
    parser.add_argument("--lm-preset", type=str, choices=["demo", "gpt"], default="demo")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--enable-arctic-shift", action="store_true")
    parser.add_argument("--merge-snippets", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1).")
    args = parser.parse_args()

    if not args.geo_dataset.exists():
        print(f"Dataset not found: {args.geo_dataset}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.geo_dataset.open("r", encoding="utf-8", newline="") as f:
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
        tasks.append({"run_id": f"{geo_id}__{qid}", "topic": topic, "geo_id": geo_id})

    if args.limit > 0:
        tasks = tasks[: args.limit]

    blocked = set(UGC_DOMAINS)
    print(f"Blocking UGC domains: {sorted(blocked)}")
    print(f"Running {len(tasks)} queries with {args.workers} workers -> {output_dir}")

    common_kw = dict(
        injection_doc_path=None,
        injection_doc_s3_uri=None,
        retriever=args.retriever,
        demo_turns=args.demo_turns,
        retrieve_top_k=args.retrieve_top_k,
        total_conv_turn=args.total_conv_turn,
        max_search_queries=args.max_search_queries,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_n,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
        lm_preset=args.lm_preset,
        skip_if_exists=not args.no_skip_existing,
        enable_arctic_shift=args.enable_arctic_shift,
        merge_snippets=args.merge_snippets,
        block_ugc_domains=blocked,
    )

    def _run_one(t):
        try:
            ok = run_single_query(
                question_id=t["run_id"],
                topic=t["topic"],
                output_dir=output_dir,
                **common_kw,
            )
            return (t["run_id"], ok, "")
        except Exception as e:
            return (t["run_id"], False, str(e))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    errors = []
    done = 0
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            run_id, ok, err = future.result()
            tag = f"[{done}/{len(tasks)}]"
            if ok:
                print(f"  {tag} {run_id} Done")
            else:
                errors.append(run_id)
                print(f"  {tag} {run_id} FAILED: {err}", file=sys.stderr)

    print(f"\nDone: {done}, Errors: {len(errors)}")
    for e in errors:
        print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
