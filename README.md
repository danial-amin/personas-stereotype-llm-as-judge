# Persona Stereotype LLM-as-Judge

This repository contains the code, notebook, and exported summaries for a persona-stereotype evaluation pipeline using 12 personas and 5 frontier LLMs, with 5 independent runs per model-persona pair.

## What Is In The Repository

Committed artifacts currently include:

- analysis code in `src/`
- runner scripts such as `run.py`, `run_human_mirror.py`, `generate_llm_summaries.py`, `analyze_mirror_distribution.py`, `export_analysis.py`, and `reparse_results.py`
- the main notebook `persona_stereotypes_analysis_v4_DA.ipynb`
- input data in `data/`
- raw and aggregated evaluation results in `results/`
- exported summaries and figures in `outputs/`

Key committed output files:

- `outputs/results_summary.txt` - human-study summary
- `outputs/llm_results_summary.txt` - aggregate 5-model LLM summary
- `outputs/mirror_results_summary.txt` - human-mirror GPT-5.4 summary
- `outputs/mirror_human_comparison.txt` - paired human vs mirror comparison
- `outputs/persona_stereo_analysis.xlsx` - exported analysis workbook
- `outputs/mirror_human_comparison.xlsx` - paired comparison workbook
- `outputs/figs/` - generated figures used in the analysis

## Results Data

Raw evaluation outputs are committed under `results/`:

- `results/<persona_id>/evaluations.json` and `evaluations.csv` — per-persona LLM runs
- `results/all_evaluations.json` and `all_evaluations.csv` — full 5-model aggregate
- `results/analysis.xlsx` — exported analysis workbook
- `results/human_mirror_gpt-5.4/` — human-mirror study outputs (510 ratings)
- `results/human_mirror_experiment/` — earlier mirror experiment run

These files support row-level recomputation of summary statistics, model-level counts, dispersion measures, and inferential analyses.

## Study Design

### LLM evaluation

- 12 personas
- 5 models
- 5 independent runs per model-persona pair
- intended total: 300 LLM evaluations

Configured model keys in `config.yaml`:

| Key | Model | Provider |
|-----|-------|----------|
| `gpt-5.4` | GPT-5.4 | OpenAI |
| `sonnet-4.6` | Claude Sonnet 4.6 | Anthropic |
| `gemini-3` | Gemini 3.1 Pro | Google |
| `grok` | Grok 4.3 | xAI |
| `qwen` | Qwen 3.6 Plus | Alibaba DashScope |

### Human mirror study

The committed mirror outputs summarize:

- 85 virtual participants
- 6 personas per participant
- intended total: 510 persona ratings
- paired comparison against the matched human-study assignment schedule

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add API keys to `.env` as needed for the providers you plan to run.

## Input Data

Personas are defined in `data/personas.csv` with these columns:

| Column | Description |
|--------|-------------|
| `persona_id` | Unique ID |
| `name` | Persona name |
| `age` | Age |
| `gender` | Gender |
| `workforce` | Occupation / industry |
| `description` | Persona text |
| `image_path` | Local path or HTTP/HTTPS image URL |

If `image_path` is a URL, images are downloaded into `data/images/` and tracked in `data/images/_download_manifest.json`.

The evaluation template lives in `prompts/evaluation_prompt.txt`.

## Running The Pipeline

New runs write artifacts under `results/`. Existing committed results are preserved unless you pass `--no-resume`.

```bash
# Download persona images referenced by URL
python download_images.py

# Force re-download of remote images
python download_images.py --refresh

# Validate inputs without model calls
python run.py --dry-run

# Smoke test a single API call
python run.py --test

# Smoke test a specific persona/model pair
python run.py --test --persona-id a_us_marcus --model gpt-5.4

# Run one persona across models
python run.py --persona-id a_us_marcus

# Run one model across personas
python run.py --model gpt-5.4

# Ignore saved run artifacts
python run.py --no-resume
```

By configuration, full LLM execution is intended to make 300 API calls.

## Expected Runtime Outputs

When you run the pipeline locally, the code writes to `results/`:

- `results/<persona_id>/evaluations.json`
- `results/<persona_id>/evaluations.csv`
- `results/all_evaluations.json`

Completed runs are resumable: existing `(model, run_index)` pairs in `results/<persona_id>/evaluations.json` are skipped by default.

## Analysis Scripts

Main analysis entry points:

- `generate_llm_summaries.py` - builds text summaries for aggregate LLM and mirror outputs
- `analyze_mirror_distribution.py` - produces paired human vs mirror comparisons
- `export_analysis.py` - exports flat analysis tables and workbooks
- `reparse_results.py` - reparses stored evaluation outputs
- `persona_stereotypes_analysis_v4_DA.ipynb` - notebook-based analysis workflow

## Configuration

`config.yaml` controls:

- `runs_per_model`
- `temperature`
- `max_tokens`
- retry settings and request delay
- provider/model identifiers
- `paths.results_dir`, which defaults to `results`

`.env` is only for secrets and provider credentials.
