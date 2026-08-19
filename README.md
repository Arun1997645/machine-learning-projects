# Spam Detection Lesson (Beginner Friendly)

This project teaches machine learning step by step using your CSV file:
`spam_ham_dataset.csv`.

The full lesson code is in:
- `scripts/run_spam_lesson.py`
- `src/spam_ml_lesson/pipeline.py`

## Folder Structure

```text
machine-learning-project/
├── spam_ham_dataset.csv
├── artifacts/
│   ├── models/
│   ├── plots/
│   └── reports/
├── data/
│   └── raw/
├── scripts/
│   └── run_spam_lesson.py
├── src/
│   └── spam_ml_lesson/
│       ├── __init__.py
│       └── pipeline.py
├── pyproject.toml
└── requirements.txt
```

## What You Will Learn

The script runs in the same order as a classroom lesson:

1. Download stopwords resource
2. Load dataset
3. Plot class distribution
4. Balance dataset
5. Clean text
6. Build word clouds
7. Tokenize and pad text
8. Build LSTM model
9. Train model
10. Evaluate model
11. Save plots, model, and metrics

## Setup

```bash
cd /Users/bti-001541/machine-learning-project
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run The Lesson

```bash
PYTHONPATH=src python3 scripts/run_spam_lesson.py
```

## Where Output Files Go

- Plots: `artifacts/plots/`
- Trained model: `artifacts/models/spam_lstm.keras`
- Metrics report: `artifacts/reports/metrics.txt`

## CSV Requirement

Your CSV must have these columns:
- `label` (values like `ham`, `spam`)
- `text` (email/message content)

The script checks for CSV in:
1. Project root: `spam_ham_dataset.csv`
2. `data/raw/spam_ham_dataset.csv`
