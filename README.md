# ARB / h2lbench — When Humans Write Like Machines

Code and experiment pipeline for **"ARB: A Matched Authorship-Rewriting Benchmark Dataset for AI-Text Detector Evaluation"** (`paper/main.tex`).

AI-text detectors are usually benchmarked on **Human vs. directly-generated-LLM-text**. This repo builds and evaluates a benchmark that separates *who came up with the content* from *who produced the final wording*, so that detector performance can be tested under LLM-mediated rewriting rather than only direct generation.

## The four-regime design

| Regime | Content origin | Linguistic surface | How it's produced |
| --- | --- | --- | --- |
| `human` | human | human-written | original sampled source text |
| `free_llm` | LLM | LLM-generated | LLM writes freely from a dataset-derived topic |
| `h2l` | human | LLM-mediated rewrite | LLM rewrites the human source text |
| `llm2l` | LLM | LLM-mediated rewrite | the same LLM rewrites its own `free_llm` output |

Each of the three source datasets (XSum, WritingPrompts, OpenWebText) contributes 600 stratified-sampled human texts (200 short/medium/long each), and every sample is carried through all four regimes for four generator models — giving matched `dataset × generator` blocks that detectors are evaluated on. Full rationale is in `paper/main.tex`, Section 3 ("ARB Benchmark Design").

## Datasets, models, detectors

| Key | Hugging Face id | Human-text field | `free_llm` topic source |
| --- | --- | --- | --- |
| `xsum` | `EdinburghNLP/xsum` | `document` | `summary` |
| `wp` | `euclaise/writingprompts` | `story` | `prompt` |
| `owt` | `Skylion007/openwebtext` | `text` | first 2 cleaned sentences (≤40 words) |

| Model key | Hugging Face id |
| --- | --- |
| `llama32_2b` (config file; internal name `llama32_3b`) | `meta-llama/Llama-3.2-3B-Instruct` |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `gemma2_9b` | `google/gemma-2-9b-it` |

| Detector key | What it is |
| --- | --- |
| `bert` | BERT-Defense supervised encoder (local checkpoint) |
| `roberta` | RoBERTa-Defense supervised encoder (local checkpoint) |
| `radar` | RADAR (`TrustSafeAI/RADAR-Vicuna-7B`), paraphrase-robust supervised detector |
| `binoculars` | Binoculars, `tiiuae/falcon-7b` observer + `tiiuae/falcon-7b-instruct` performer, zero-shot |
| `fastdetectgpt` | FastDetectGPT, zero-shot, sampling/scoring model configurable via env vars |

Generation decoding is fixed across all models/regimes: `temperature=0.7`, `top_p=0.9`, `top_k=40`, `max_new_tokens=512`, seed `42` (see `conf/experiment/main.yaml`).

## Repository layout

```
conf/                  Hydra configuration (dataset, model, detector, experiment groups)
prompts/                System + regime prompt templates (system.py, freellm.py, h2l.py, llm2lm.py)
utils/                  Core library: dataset loading, filtering, sampling, generation, detectors, metrics
exploratory/            Generation scripts actually used to produce the benchmark texts (run_freellm.py, run_generate.py, run_llm2l.py) + notebooks
scripts/                Batch orchestration (batch_all.sh), h2l realignment, figure generation
dataset_generation.py   Hydra CLI: load + clean/filter a raw HF dataset split
run_detector.py         Hydra CLI: score generated texts with a detector, save metrics
evaluate_results.py     Aggregate results-aligned/*.npz into the paper's summary tables
print_metrics.py        Quick console dump of all saved metrics
generated-texts/        Sampled human texts + raw generated regime texts (*.txt, gitignored)
generated-texts-aligned/ H2L texts realigned to canonical HF sample order
results/, results-aligned/  Per (detector, dataset, model, regime) metrics (*.npz)
ARB-Dataset/            HF-Hub-ready release of the benchmark (separate git repo/submodule)
paper/                  LaTeX source of the paper (main.tex)
tests/                  pytest tests (alignment, text analysis)
```

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

All commands below assume `./venv/bin/python` and are run **from the repository root** unless stated otherwise.

You'll also need, depending on which detectors/models you run:

- `export HF_TOKEN=...` — required to pull gated models (Llama, FastDetectGPT's Llama3-8B sampling/scoring models).
- BERT-Defense and RoBERTa-Defense checkpoints available locally at `~/.pretrained-models/BERT-Defense` and `~/.pretrained-models/RoBERTa-Defense` (see `conf/detector/bert.yaml` / `roberta.yaml`).
- A local HF datasets cache at `~/.datasets` (`conf/experiment/main.yaml: dataset_home_path`).
- A CUDA GPU for anything beyond the smallest models — the paper's experiments used a single 32GB GPU with 8-bit quantization for FastDetectGPT.

## 1. Generate texts (XSum / WritingPrompts / OpenWebText)

### 1.1 Preprocess the source dataset

Filters a raw Hugging Face split (word-length bounds, artifact/toxicity/acceptability filters) and saves the cleaned dataset:

```bash
./venv/bin/python dataset_generation.py dataset=xsum   # or dataset=wp / dataset=owt
```

### 1.2 Stratified sampling

Draw the seeded 600-per-dataset sample (200 short / 200 medium / 200 long, `utils/sampling.stratified_random_sampling_by_range`, seed 42), producing a Hugging Face dataset at `generated-texts/<dataset>_sampled`. This step is currently driven from `exploratory/datasets.ipynb`, which calls `add_range_to_dataset` (`utils/range_classification.py`) and `stratified_random_sampling_by_range` (`utils/sampling.py`) on the cleaned dataset from step 1.1.

### 1.3 Generate the three machine regimes

Run from `exploratory/` (each script loads Hydra config from `../conf` and writes to `output/<dataset>_<model>_<regime>.txt`; copy/symlink these into `generated-texts/` before running detectors, matching `<dataset>_<model>_<regime>.txt`):

```bash
cd exploratory

# free_llm — LLM writes freely from the dataset-derived topic
../venv/bin/python run_freellm.py dataset=xsum model=llama32_2b
../venv/bin/python run_freellm.py dataset=xsum model=qwen25_7b
../venv/bin/python run_freellm.py dataset=xsum model=mistral7b
../venv/bin/python run_freellm.py dataset=xsum model=gemma2_9b
# repeat with dataset=wp and dataset=owt

# h2l — LLM rewrites the human source text
../venv/bin/python run_generate.py dataset=xsum model=llama32_2b
# repeat per model/dataset

# llm2l — same LLM rewrites its own free_llm output (needs the free_llm output first)
../venv/bin/python run_llm2l.py dataset=xsum model=llama32_2b
# repeat per model/dataset
```

Prompts are defined in `prompts/freellm.py`, `prompts/h2l.py`, `prompts/llm2lm.py`, with the shared system prompt in `prompts/system.py`. These are reproduced verbatim in `paper/main.tex` (Appendix, "Prompt Templates") and in `ARB-Dataset/README.md`.

### 1.4 Realign H2L ordering (if needed)

`h2l` rows can drift out of canonical HF sample order; `scripts/realign_h2l.py` recovers the correct alignment via exact/fuzzy text matching and writes corrected files + results to `generated-texts-aligned/` and `results-aligned/`:

```bash
./venv/bin/python scripts/realign_h2l.py --generated-dir generated-texts --results-dir results \
    --aligned-generated-dir generated-texts-aligned --aligned-results-dir results-aligned
```

## 2. Run detectors against generated texts

`run_detector.py` builds a `human` vs. `<regime>` dataset from `generated-texts/<dataset>_sampled` + the matching machine-text file, scores it with the selected detector, computes metrics (AUROC, AUPRC, F1, precision, recall, TPR@1%FPR, ECE, Brier), and saves them to `results-aligned/<detector>_<dataset>_<model>_<regime>.npz`.

```bash
./venv/bin/python run_detector.py \
    dataset=xsum \
    model=llama32_2b \
    detector=binoculars \
    machine_postfix=h2l
```

- `machine_postfix` selects the regime to compare against `human`: `free_llm` (aliases `freellm`, `llmfree`), `h2l`, or `llm2l`.
- `detector` ∈ `bert`, `roberta`, `radar`, `binoculars`, `fastdetectgpt`.
- `dataset` ∈ `xsum`, `wp`, `owt`. `model` ∈ `llama32_2b`, `qwen25_7b`, `mistral7b`, `gemma2_9b`.
- Re-running with the same arguments is a no-op if the metrics file already exists.

### Batch grid

`scripts/batch_all.sh` queues the full `dataset × model × detector × regime` grid via [task-spooler](https://github.com/justanhduc/task-spooler) (`tsp`):

```bash
./scripts/batch_all.sh
```

## 3. Aggregate results & reproduce paper tables

```bash
./venv/bin/python evaluate_results.py     # baseline / h2l robustness / llm2l robustness / source-origin-gap tables -> results-aligned/*.csv
./venv/bin/python print_metrics.py        # quick console dump of all saved metrics
```

### Figures

```bash
make gen-images          # heatmaps of H2L AUROC / TPR@1%FPR (scripts/generate_figure_heatmap_h2l_*.py)
```

Other figure scripts (`scripts/generate_figure1_delta_tpr_h2l.py`, `generate_figure2_delta_tpr_llm2l.py`, `generate_figure3_detection_gap.py`, `generate_figure_textual_rewriting_paths.py`) can be run directly with `./venv/bin/python scripts/<name>.py`.

## Hydra configuration reference

Everything is composed from `conf/config.yaml` (defaults: `experiment=main`, `dataset=xsum`, `detector=roberta`, `model=llama32_2b`), overridable on the command line as `key=value`:

- `conf/dataset/{xsum,wp,owt}.yaml` — source dataset identifiers, fields, topic source.
- `conf/model/{llama32_2b,qwen25_7b,mistral7b,gemma2_9b}.yaml` — generator model id and precision.
- `conf/detector/{bert,roberta,radar,binoculars,fastdetectgpt}.yaml` — detector implementation and settings.
- `conf/experiment/main.yaml` — paths, length strata/ranges, regimes, generation decoding params, evaluation metrics and bootstrap settings.
- `conf/experiment/pilot.yaml` — small pilot run (`is_pilot=true`).
- `conf/experiment/sensitivity_temperature.yaml` — temperature-sensitivity sweep over `[0.2, 0.7, 1.0]` for `h2l`.

Use `experiment=pilot` to smoke-test the pipeline on ~100 samples before a full run.

## Publishing the benchmark to Hugging Face

`ARB-Dataset/` is the release-ready version of the benchmark (separate git history, published as `giper45/ARB-Dataset`). Rebuild the parquet release from `generated-texts/` + `generated-texts-aligned/`:

```bash
cd ARB-Dataset
../venv/bin/python scripts/build_hf_dataset.py
```

See `ARB-Dataset/README.md` for the dataset card, field documentation, and regime definitions as published on the Hub.

## Tests

```bash
./venv/bin/python -m pytest tests/
```

