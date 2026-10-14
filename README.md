# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 individual midterm project.

## Files
- `project.py`: CNN, BiLSTM, and analysis experiments.
- `make_report.py`: report generator.
- `Project_Report.pdf`: research report (replace `[GitHub URL]` on page 7 with the actual repository URL).
- `metrics_summary.csv`: supplied three-seed experiment metrics.
- `reflections.md`: reflections used for the report.
- `requirements.txt`: dependencies.
- `Finish_CMPSC448_Project.ipynb`: Colab runner.

## Dataset
LMSYS Chatbot Arena Conversations: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
The dataset is **not redistributed** here. Obtain access under the dataset's terms, preprocess as documented in the report, and place the prepared file at `data/chatbot_arena.parquet`.

## Reproduce
```bash
pip install -r requirements.txt
python project.py --data data/chatbot_arena.parquet --out results --seeds 42 43 44 --rq3
python make_report.py --team "Victor Madu (individual project)"
```

Note: `metrics_summary.csv` is provided, but the full raw experiment artifacts are not included in this package. The report's additional figures and analyses were not independently reproduced when this repository package was assembled.
