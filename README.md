# GEO-STORM: Measuring Poisoning Vulnerabilities in Deep-Research Agents

Code for the paper *"GEO-STORM: Measuring Poisoning Vulnerabilities in Deep-Research Agents"*.

## Overview

We evaluate how easily adversarial content injected into user-generated content (UGC) platforms (Reddit, Quora, etc.) propagates through deep-research agents that retrieve, synthesize, and cite web sources. We study five systems:

- **Co-STORM** (collaborative multi-turn research, open-source)
- **STORM** (multi-perspective wiki generation, open-source)
- **OmniThink** (mind-map-guided deep research, open-source)
- **Gemini Deep Research** (Google, proprietary)
- **OpenAI Deep Research** (OpenAI, proprietary)

We implement two attack vectors:

1. **SERP-snippet attack**: append a short (~13-word) adversarial text to search-engine result page snippets via UGC-mimic injection (simulating SEO manipulation of snippet previews).
2. **Full-content attack**: inject longer (~130-word) adversarial passages into full Reddit thread content retrieved via Arctic Shift archives.

For each attack, we test three targeting strategies: single-URL, multi-URL (3 URLs), and domain-prefix matching.

## Repository Structure

```
geo_storm/
├── knowledge_storm/          # Modified STORM/Co-STORM engine (from stanford-oval/storm)
│   ├── storm_wiki/           #   STORM pipeline modules
│   ├── collaborative_storm/  #   Co-STORM pipeline modules
│   ├── rm.py                 #   Retrieval models (Serper, Bing, You.com)
│   └── utils.py              #   WebPageHelper with Reddit JSON API support
├── omnithink/                # Modified OmniThink engine
│   └── src/                  #   Core OmniThink modules
├── examples/
│   ├── batch/                # Co-STORM core: single-query runner + UGC-mimic injector
│   ├── costorm_batch/        # Co-STORM batch runner (--dataset, --ugc-config)
│   ├── storm_batch/          # STORM batch runner (--dataset, --ugc-config)
│   ├── omnithink_batch/      # OmniThink batch runner (--dataset, --ugc-config)
│   ├── gemini_dr_batch/      # Gemini Deep Research batch runner
│   ├── openai_dr_batch/      # OpenAI Deep Research batch runner
│   ├── geo_examples/         # GEO dataset construction and evaluation
│   └── manual_examples/      # Manual document injection helper
├── experiments/
│   ├── serp_main_13w/        # Main SERP-snippet experiment (13-word payloads)
│   │   ├── configs/          #   9 UGC configs (3 systems x 3 strategies)
│   │   ├── build_strict_configs.py
│   │   └── evaluate_all.py
│   └── serp_length_ablation/ # Adversarial text length ablation (10-40 words)
│       └── configs/          #   Length-varied configs per system
├── runners/                  # Orchestration utilities
│   └── run_remaining_dr_clean.py     # Orchestrator for Gemini/OpenAI DR reruns
├── analysis/                 # Recon analysis and verification
│   ├── analyze_storm_ugc.py          # UGC URL analysis for STORM clean runs
│   ├── analyze_omnithink_ugc.py      # UGC URL analysis for OmniThink clean runs
│   ├── analyze_cited_ugc_all_systems.py  # Cross-system cited UGC comparison
│   └── verify_recon_tables.py        # Verify recon table numbers
├── config_builders/          # UGC config generation from clean run data
├── geo_out/                  # Dataset CSVs (no run outputs)
├── evaluate_ugc.py           # Main evaluation: exposure, citation, mention rates
├── secrets.toml.template     # API key template
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- API keys: OpenAI (`gpt-4o` for LLM calls, `o4-mini-deep-research` for OpenAI DR), Google/Gemini (`deep-research-pro-preview` for Gemini DR), Serper (for web search)

### Installation

```bash
# Clone and enter the repository
cd geo_storm

# Create conda environment
conda create -n storm python=3.10 -y
conda activate storm

# Install dependencies
pip install -r requirements.txt

# For OmniThink experiments (separate environment)
conda create -n OmniThink python=3.10 -y
conda activate OmniThink
pip install -r omnithink/requirements.txt

# Configure API keys
cp secrets.toml.template secrets.toml
# Edit secrets.toml with your OPENAI_API_KEY and SERPER_API_KEY
```

## Running Experiments

### 1. Reconnaissance: Clean Baseline Runs

Run each system without any attack to discover which UGC URLs appear in search results:

```bash
# Co-STORM clean runs
python -m examples.costorm_batch.run_costorm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/costorm/clean_runs --workers 4

# STORM clean runs
python -m examples.storm_batch.run_storm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/storm/clean_runs --workers 4

# OmniThink clean runs (requires OmniThink conda env)
conda run -n OmniThink python -m examples.omnithink_batch.run_omnithink_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/omnithink/clean_runs --workers 2

# Gemini Deep Research clean runs
python -m examples.gemini_dr_batch.run_gemini_dr_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/gemini_dr/clean_runs

# OpenAI Deep Research clean runs
python -m examples.openai_dr_batch.run_openai_dr_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/openai_dr/clean_runs
```

### 2. Analyze Recon Results and Build Attack Configs

Extract recurring UGC URLs from clean runs and generate attack configurations:

```bash
# Analyze UGC URLs across systems
python analysis/analyze_cited_ugc_all_systems.py

# Build attack configs for all 3 systems x 3 strategies
conda run -n storm python experiments/serp_main_13w/build_strict_configs.py
```

This produces 9 JSON config files in `experiments/serp_main_13w/configs/`, one per (system, strategy) pair. Each config maps question IDs to target URLs and compressed adversarial text.

### 3. SERP-Snippet Attack

Run the SERP-snippet attack for each system and strategy:

```bash
# Co-STORM (1-URL strategy)
python -m examples.costorm_batch.run_costorm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir experiments/serp_main_13w/runs/costorm_1url \
    --ugc-config experiments/serp_main_13w/configs/ugc_config_costorm_1url_15w.json \
    --ugc-append-mode --workers 4

# STORM (1-URL strategy)
python -m examples.storm_batch.run_storm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir experiments/serp_main_13w/runs/storm_1url \
    --ugc-config experiments/serp_main_13w/configs/ugc_config_storm_1url_15w.json \
    --ugc-append-mode --workers 4

# OmniThink (1-URL strategy, requires OmniThink conda env)
conda run -n OmniThink python -m examples.omnithink_batch.run_omnithink_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir experiments/serp_main_13w/runs/omnithink_1url \
    --ugc-config experiments/serp_main_13w/configs/ugc_config_omnithink_1url_15w.json \
    --ugc-append-mode --workers 2

# Repeat with --ugc-config .../ugc_config_<system>_3url_15w.json for 3-URL strategy
# Repeat with --ugc-config .../ugc_config_<system>_domain_15w.json for domain strategy
```

### 4. Full-Content Attack (Arctic Shift)

Run the full-content attack using archived Reddit data. Same batch runners, with `--enable-arctic-shift`:

```bash
# Co-STORM full-content
python -m examples.costorm_batch.run_costorm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/costorm/fullcontent_3url_runs \
    --ugc-config geo_out/costorm/ugc_config_3url.json \
    --ugc-append-mode --enable-arctic-shift --merge-snippets --workers 4

# STORM full-content
python -m examples.storm_batch.run_storm_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/storm/fullcontent_3url_runs \
    --ugc-config geo_out/storm/ugc_config_3url.json \
    --ugc-append-mode --enable-arctic-shift --merge-snippets --workers 4

# OmniThink full-content (requires OmniThink conda env)
conda run -n OmniThink python -m examples.omnithink_batch.run_omnithink_batch \
    --dataset geo_out/geo_dataset_clean.csv \
    --output-dir geo_out/omnithink/fullcontent_3url_runs \
    --ugc-config geo_out/omnithink/ugc_config_3url.json \
    --ugc-append-mode --enable-arctic-shift --workers 2
```

### 5. Evaluation

Evaluate attack success rates (exposure, citation, mention). The same `evaluate_ugc.py` works for all systems and both attack types:

```bash
# Evaluate SERP-snippet attack
python evaluate_ugc.py \
    --runs-dir experiments/serp_main_13w/runs/costorm_1url \
    --dataset geo_out/geo_dataset_ugc_aware.csv

# Evaluate full-content attack
python evaluate_ugc.py \
    --runs-dir geo_out/costorm/fullcontent_3url_runs \
    --dataset geo_out/geo_dataset_ugc_aware.csv
```

### 6. Length Ablation

```bash
# Run SERP-snippet ablation across payload lengths (10, 15, 20, 25, 30, 40 words)
conda run -n storm python experiments/serp_length_ablation/run_ablation.py
python experiments/serp_length_ablation/evaluate_ablation.py
```

## Key Metrics

- **Exposure (E)**: fraction of runs where the adversarial text was fetched by the system
- **Citation | Exposure (C|E)**: among exposed runs, fraction where the target entity appears in the knowledge base / cited sources
- **Mention | Exposure (M|E)**: among exposed runs, fraction where the target entity appears in the final generated report
- **Adversarial ratio**: median ratio of injected adversarial words to all words returned by the search engine

## Modifications from Upstream

This codebase is based on [stanford-oval/storm](https://github.com/stanford-oval/storm) (STORM/Co-STORM) and [zjunlp/OmniThink](https://github.com/zjunlp/OmniThink). Key modifications:

- **UGC-mimic injection** (`examples/batch/_injector.py`): `UGCMimicRetriever` wraps the base retriever to append adversarial text to matching SERP snippets or full-content results
- **Reddit JSON API** (`knowledge_storm/utils.py`): bypass anti-scraper blocking by fetching Reddit content via the public `.json` API endpoint
- **Batch orchestration**: parallel runners for all three systems with skip-if-exists logic
- **Arctic Shift integration**: full-content injection using archived Reddit thread data

## Dataset

The `geo_out/` directory contains the experiment datasets (CSV only, no run outputs):

- `geo_dataset_clean.csv`: 176 queries across 11 product clusters, used for clean baseline runs
- `geo_dataset_ugc_aware.csv`: same queries augmented with adversarial documents and UGC metadata

## License

This project is licensed under the MIT License (see `LICENSE`). It includes components from [stanford-oval/storm](https://github.com/stanford-oval/storm) (MIT, see `LICENSE`) and [zjunlp/OmniThink](https://github.com/zjunlp/OmniThink) (MIT, see `omnithink/LICENSE`).
