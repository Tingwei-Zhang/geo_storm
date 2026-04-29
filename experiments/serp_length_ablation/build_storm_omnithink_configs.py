#!/usr/bin/env python3
"""
Build per-length UGC configs for STORM and OmniThink using existing
system-specific configs (which have per-system target_urls) and
replacing the adversarial_text with compressed versions from the
Co-STORM configs.

The compressed texts are system-agnostic (same content, just shorter),
so we reuse them but keep each system's target_url assignments.
"""
import json
import sys
from pathlib import Path

CLUSTER = "comcast_xfinity_cancel"
LENGTHS = ["10", "15", "20", "25", "30", "40"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COSTORM_CONFIGS = Path(__file__).resolve().parent / "configs"
STORM_ORIG = PROJECT_ROOT / "geo_out" / "storm" / "ugc_config_1url.json"
OMNITHINK_ORIG = PROJECT_ROOT / "geo_out" / "omnithink" / "ugc_config_1url.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "configs"


def build_system_configs(system_name: str, orig_config_path: Path):
    with open(orig_config_path) as f:
        orig = json.load(f)

    orig_rules = orig.get("rules_by_question_id", {})
    comcast_rules = {k: v for k, v in orig_rules.items() if CLUSTER in k}
    print(f"{system_name}: {len(comcast_rules)} comcast entries")

    # Original-length config (just comcast entries)
    out = {"separator": "", "rules_by_question_id": comcast_rules}
    out_path = OUTPUT_DIR / f"ugc_config_{system_name}_comcast_original.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote {out_path.name}")

    # Per-length configs: use compressed text from Co-STORM configs
    for length in LENGTHS:
        costorm_path = COSTORM_CONFIGS / f"ugc_config_comcast_{length}w.json"
        with open(costorm_path) as f:
            costorm_config = json.load(f)
        costorm_rules = costorm_config["rules_by_question_id"]

        new_rules = {}
        for qid, rule in comcast_rules.items():
            if qid in costorm_rules:
                new_rules[qid] = {
                    "target_url": rule.get("target_url", ""),
                    "target_urls": rule.get("target_urls", []),
                    "domain_prefixes": rule.get("domain_prefixes", []),
                    "adversarial_text": costorm_rules[qid]["adversarial_text"],
                }
            else:
                print(f"  WARNING: {qid} not in Co-STORM {length}w config, skipping")

        out = {"separator": "", "rules_by_question_id": new_rules}
        out_path = OUTPUT_DIR / f"ugc_config_{system_name}_comcast_{length}w.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Wrote {out_path.name} ({len(new_rules)} entries)")


def main():
    build_system_configs("storm", STORM_ORIG)
    print()
    build_system_configs("omnithink", OMNITHINK_ORIG)
    print("\nDone.")


if __name__ == "__main__":
    main()
