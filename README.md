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

---

## ==================== SMS Spam Lesson ====================

This repo now also includes a second beginner-friendly use case for SMS spam detection.

## SMS Spam Lesson

Files for the SMS lesson:
- `notebooks/spam_sms_step_by_step.ipynb`
- `scripts/run_spam_sms_lesson.py`
- `src/spam_sms_lesson/pipeline.py`

The SMS lesson keeps the code very simple and explains each step clearly.

Supported SMS CSV formats:
- `label` and `text`
- `v1` and `v2` (common public SMS spam dataset format)

The SMS script checks for a dataset in these locations:
1. Project root: `spam_sms_dataset.csv`
2. Project root: `sms_spam_dataset.csv`
3. Project root: `spam.csv`
4. `data/raw/spam_sms_dataset.csv`
5. `data/raw/sms_spam_dataset.csv`
6. `data/raw/spam.csv`

Run the SMS lesson with:

```bash
PYTHONPATH=src python3 scripts/run_spam_sms_lesson.py
```

SMS output files go to:
- `artifacts/models/spam_sms_lstm_from_script.keras`
- `artifacts/reports/spam_sms_metrics_from_script.txt`
- `artifacts/models/spam_sms_lstm_from_notebook.keras`
- `artifacts/reports/spam_sms_metrics_from_notebook.txt`

## CSV Requirement

Your SMS CSV must have these columns:
- `label` (values like `ham`, `spam`)
- `text` (the SMS message content)

For the SMS lesson, the notebook and script look for `spam.csv` in:
1. The project root
2. `data/raw/`
