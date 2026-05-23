# Persona Stereotype LLM-as-Judge

Evaluate 12 personas for stereotypical content using 5 frontier LLMs. Each model runs **5 independent evaluations** per persona (no shared context between runs) to measure consistency.

## Models

| Key | Model | Provider |
|-----|-------|----------|
| `gpt-5.4` | GPT-5.4 | OpenAI |
| `sonnet-4.6` | Claude Sonnet 4.6 | Anthropic |
| `gemini-3` | Gemini 3.1 Pro Preview | Google |
| `deepseek` | DeepSeek V4 Pro | DeepSeek |
| `grok` | Grok 4.3 | xAI |

Model IDs are configurable in `config.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
```

## Personas

Edit `data/personas.csv` with your 12 personas:

| Column | Description |
|--------|-------------|
| `persona_id` | Unique ID (e.g. `persona_01`) |
| `name` | Persona name |
| `age` | Age |
| `gender` | Gender |
| `workforce` | Occupation / industry |
| `description` | Full persona text |
| `image_path` | Path to persona image (relative or absolute) |

Place images in `data/images/`.

## Evaluation prompt

The fixed prompt template lives in `prompts/evaluation_prompt.txt`. Persona fields are injected via `{name}`, `{age}`, `{gender}`, `{workforce}`, and `{description}`. Edit this file to change the rubric or output schema.

## Run

```bash
# Full run (12 personas × 5 models × 5 runs = 300 API calls)
python run.py

# Dry run — validate CSV/images without calling APIs
python run.py --dry-run

# Single persona
python run.py --persona-id persona_01

# Single model
python run.py --model gpt-5.4

# Re-run from scratch (ignore saved results)
python run.py --no-resume
```

Runs are **resumable** by default: completed `(model, run_index)` pairs are skipped if `results/<persona_id>/evaluations.json` already exists.

## Output

Per persona (`results/<persona_id>/`):

- `evaluations.json` — full structured results including raw responses
- `evaluations.csv` — flattened table for analysis

Aggregate:

- `results/all_evaluations.json` — all personas in one file

Each evaluation row includes parsed fields (`contains_stereotype`, `stereotype_severity`, scores, `reasoning`, etc.) when the model returns valid JSON.

## Configuration

See `config.yaml` for:

- `runs_per_model` (default: 5)
- `temperature` (default: 0.0)
- `max_tokens`, retry settings, rate-limit delay
- Model IDs and display names

## Cost note

A full run makes **300 API calls** (12 personas × 5 models × 5 runs). Use `--persona-id` and `--model` to test incrementally first.
