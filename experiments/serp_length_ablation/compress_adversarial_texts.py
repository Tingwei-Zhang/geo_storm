#!/usr/bin/env python3
"""
Step 1: Compress adversarial texts to target word lengths using GPT.

Reads the original ugc_config_1url.json, extracts comcast_xfinity_cancel entries,
and compresses each adversarial_text to ~10, 15, 20, 25, 30, 40 words.
Outputs one ugc_config per length variant.
"""
import json
import os
import sys
from pathlib import Path

import toml
from openai import OpenAI

CLUSTER = "comcast_xfinity_cancel"
TARGET_NAME = "CancelEase"
TARGET_LENGTHS = [10, 15, 20, 25, 30, 40]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = PROJECT_ROOT / "secrets.toml"
ORIGINAL_CONFIG = PROJECT_ROOT / "geo_out" / "costorm" / "ugc_config_1url.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "configs"


def load_api_key():
    secrets = toml.load(SECRETS_PATH)
    return secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def compress_text(client: OpenAI, original: str, target_words: int) -> str:
    prompt = (
        f"Compress the following text to approximately {target_words} words. "
        f"Keep the key product/entity name '{TARGET_NAME}' and the core message. "
        f"Output ONLY the compressed text, nothing else.\n\n"
        f"Original text:\n{original}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main():
    api_key = load_api_key()
    if not api_key:
        print("No OPENAI_API_KEY found", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    with open(ORIGINAL_CONFIG, "r") as f:
        config = json.load(f)

    rules = config["rules_by_question_id"]
    comcast_rules = {k: v for k, v in rules.items() if CLUSTER in k}
    print(f"Found {len(comcast_rules)} entries for {CLUSTER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_compressed = {}

    for target_len in TARGET_LENGTHS:
        print(f"\n--- Compressing to ~{target_len} words ---")
        compressed_rules = {}

        for qid, rule in comcast_rules.items():
            original = rule["adversarial_text"]
            compressed = compress_text(client, original, target_len)
            actual_wc = len(compressed.split())
            print(f"  {qid}: {len(original.split())} -> {actual_wc} words")
            compressed_rules[qid] = {
                "target_url": rule["target_url"],
                "adversarial_text": compressed,
            }

        variant_config = {
            "separator": "",
            "rules_by_question_id": compressed_rules,
        }
        out_path = OUTPUT_DIR / f"ugc_config_comcast_{target_len}w.json"
        with open(out_path, "w") as f:
            json.dump(variant_config, f, indent=2)
        print(f"  Wrote {out_path}")
        all_compressed[target_len] = compressed_rules

    # Also create the original-length config (just comcast entries)
    original_config = {
        "separator": "",
        "rules_by_question_id": comcast_rules,
    }
    out_path = OUTPUT_DIR / "ugc_config_comcast_original.json"
    with open(out_path, "w") as f:
        json.dump(original_config, f, indent=2)
    print(f"\nWrote original config: {out_path}")

    # Summary
    summary_path = OUTPUT_DIR / "compression_summary.json"
    summary = {}
    for target_len in TARGET_LENGTHS:
        lengths = [
            len(compressed_rules[qid]["adversarial_text"].split())
            for qid, compressed_rules in [(qid, all_compressed[target_len]) for qid in comcast_rules]
        ]
        summary[f"{target_len}w"] = {
            "target": target_len,
            "actual_mean": sum(lengths) / len(lengths) if lengths else 0,
            "actual_min": min(lengths) if lengths else 0,
            "actual_max": max(lengths) if lengths else 0,
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
