#!/usr/bin/env python3
"""
Analyze UGC URLs from Gemini Deep Research runs.

Same analysis as analyze_openai_dr.py — the references.json format is identical
(url_to_unified_index, url_to_info).

Usage:
    cd geo_storm
    python -m examples.gemini_dr_batch.analyze_gemini_dr \
        --dataset geo_out/geo_dataset_clean.csv \
        --runs-dir geo_out/gemini_dr/clean_runs \
        --output geo_out/gemini_dr/recurring_ugc_urls.csv
"""
import sys
from pathlib import Path

# Add project root so we can import the shared analysis
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.openai_dr_batch.analyze_openai_dr import main

if __name__ == "__main__":
    main()
