# Who Wrote It? Identifying LLMs from Their Responses
**CMPSC 448 midterm** - CNN and BiLSTM classifiers for LLM-family attribution (RQ1, RQ2, plus optional RQ3 and RQ4).

## Data (not redistributed)
Recommended: LMSYS *Chatbot Arena Conversations* (gated; accept terms at
https://huggingface.co/datasets/lmsys/chatbot_arena_conversations). Save the parquet as `data/chatbot_arena.parquet`.
Prompts are CC BY 4.0, model outputs CC BY-NC 4.0. Do **not** commit raw data (see `.gitignore`); review it for sensitive text.
Your own data also works: a `.csv/.jsonl/.parquet` with columns `LLM_name,LLM_Input,LLM_output` (optional `question_id`).
The dataset is from 2023 and covers GPT, Claude, Llama-derived, ChatGLM and Google models only.

## Run
```bash
python -m pip install -r requirements.txt
python project.py --data data/chatbot_arena.parquet --out results --seeds 42 43 44 --rq3   # full run
python project.py --data data/chatbot_arena.parquet --max-per-family 150 --min-per-family 75 --epochs 2   # quick check
# lessons are in reflections.md; rebuild the PDF from results/ with:
python make_report.py --team "Victor Madu (individual project)" --results results   # builds Project_Report.pdf from results/
```
`make_report.py` pulls every table, figure and data-driven sentence from `results/`, so the PDF always matches the experiments.
Before you run anything it produces a template with `[RUN REQUIRED]` markers.

## What the pipeline does
- Builds (LLM_name, LLM_Input, LLM_output) rows: English only, moderation-flagged conversations dropped, ambiguous names
  (e.g. GPT4All) excluded, exact duplicate responses removed, families capped for balance.
- **Leakage control:** stratified 70/10/20 split grouped by `question_id` (asserted), train-only vocabulary, test used once.
- Conditions: input / output / input+output (prompt capped at 1/3 of the window) and `output_noformat` (formatting stripped; RQ4 ablation).
- Models: TextCNN (masked max-pool) and BiLSTM (packed sequences); AdamW, early stopping on validation macro-F1.
- Baselines: chance, majority class, TF-IDF + logistic regression. Uncertainty: multi-seed mean/std and group-bootstrap 95% CIs.
- RQ4: eight interpretable features, feature-only gradient-boosting classifier, permutation importance, formatting ablation.
- RQ3 (`--rq3`): leave-one-domain-out with keyword-based domain labels (heuristic, not ground truth).

## Validity notes
Labels are families, not checkpoints. Results describe this historical dataset, not current models. Keyword-based domains in RQ3 are noisy.
Row counts depend on your Arena download; always take numbers from `results/`.

## Included results
`results/` holds the CSV tables, figures and confusion matrices behind `Project_Report.pdf` (model weights and raw data are not included).
- `metrics.csv` / `metrics_summary.csv`: 8 conditions x 2 models x seeds 42, 43, 44 (mean and std in the summary).
- `baselines.csv`: chance, majority class, TF-IDF + logistic regression. `rq3.csv`: leave-one-domain-out (seed 42 only).
- `rq4_*.csv/png`: interpretable-feature analysis. Training curves, confusion matrices and classification reports are from seed 42.

How they were produced: seed 42 came from `project.py` (all conditions, baselines, RQ4). Seeds 43 and 44 were run as separate jobs with
`--skip-baselines` after a long run was interrupted, and the last three seed-44 conditions were finished with `run_missing_s44.py`.
`run_rq3_only.py` ran RQ3 from the saved seed-42 checkpoints (which reproduced the reported test accuracies exactly). The per-seed
metrics files were then concatenated into `results/metrics.csv`. A single `python project.py ... --seeds 42 43 44 --rq3` run reproduces everything in one go.
The 95% CIs in the report come from the seed-42 predictions only.

## Report
`Project_Report.pdf` is built by `make_report.py` from `results/` and `reflections.md`.
