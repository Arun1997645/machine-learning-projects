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

---

## ==================== Text Classification with Naive Bayes ====================

This repo now also includes a beginner-friendly text classification use case using Multinomial Naive Bayes.

## Naive Bayes Lesson

Files for the Naive Bayes lesson:
- `notebooks/text_classification_naive_bayes_step_by_step.ipynb`
- `scripts/run_text_classification_nb_lesson.py`
- `src/text_classification_nb_lesson/pipeline.py`
- `data/raw/synthetic_text_data.csv`

The Naive Bayes lesson keeps the code very simple and explains each step clearly.

What this lesson teaches:
1. Load a text classification dataset
2. Split text and labels
3. Create training and testing sets
4. Convert text into numeric features using `CountVectorizer`
5. Train a `MultinomialNB` model
6. Predict categories for test data
7. Evaluate the model using accuracy, classification report, and confusion matrix
8. Predict one new unseen sentence

Run the Naive Bayes lesson with:

```bash
PYTHONPATH=src python3 scripts/run_text_classification_nb_lesson.py
```

Naive Bayes output files go to:
- `artifacts/plots/text_nb_confusion_matrix.png`
- `artifacts/reports/text_nb_metrics_from_script.txt`

CSV requirement for this lesson:
- `text` column: the short sentence or document
- `label` column: the category name

The provided synthetic dataset includes these categories:
- `Sports`
- `Technology`
- `Politics`
- `Entertainment`

---

## ==================== Handwritten Digit Classification with TensorFlow ====================

This repo now also includes a beginner-friendly lesson that teaches a neural network
to **read handwritten digits (0–9)** using the famous MNIST dataset and TensorFlow/Keras.

## Handwritten Digits Lesson

Files for this lesson:
- `notebooks/handwritten_digits_tensorflow_step_by_step.ipynb`

This lesson is designed so that even a complete beginner can follow along.
Every step is explained in plain English, and every result is interpreted so you
know what it means.

---

## What You Will Learn (Step by Step)

| Step | What happens                          | Key idea                                        |
|------|---------------------------------------|-------------------------------------------------|
| 1    | Load the MNIST dataset                | 60,000 training images + 10,000 test images     |
| 2    | Explore class distribution            | Every digit (0–9) has roughly equal examples    |
| 3    | Visualise sample images               | Each image is 28 × 28 grayscale pixels          |
| 4    | Normalise pixel values                | Divide by 255 so values are between 0.0 and 1.0 |
| 5    | Build a Sequential neural network     | Flatten → Dense(128) → Dense(128) → Dense(10)   |
| 6    | Compile and train the model           | Adam optimiser, 5 training epochs               |
| 7    | Plot training history                 | Accuracy goes up, loss goes down = healthy      |
| 8    | Evaluate on test data                 | ~97–98 % accuracy on 10,000 unseen images       |
| 9    | Confusion matrix                      | See exactly which digits get mixed up           |
| 10   | Display correct and wrong predictions | Visual comparison of hits and misses            |
| 11   | Save and reload the trained model     | `.keras` format keeps everything intact         |
| 12   | Predict a single new image            | Confidence bar chart shows model certainty      |

---

## What is MNIST?

MNIST is one of the most famous datasets in machine learning.
It contains 70,000 black-and-white photos of handwritten digits,
already split into 60,000 for training and 10,000 for testing.
The dataset is built into Keras — no manual download needed.

---

## How to Run

Open the notebook in VS Code and run all cells from top to bottom:

```
notebooks/handwritten_digits_tensorflow_step_by_step.ipynb
```

No script file is needed — the notebook is self-contained and runs everything.

---

## Output Files

| File | Description |
|------|-------------|
| `artifacts/models/handwritten_digit_model.keras` | Trained neural network saved to disk |
| `artifacts/plots/digit_training_history.png`     | Accuracy and loss curves over epochs |
| `artifacts/plots/digit_confusion_matrix.png`     | Confusion matrix heatmap             |
| `artifacts/reports/digit_metrics.txt`            | Accuracy and classification report   |

---

## Neural Network Architecture

```
Input image  (28 × 28 pixels)
        ↓
Flatten  →  converts 28×28 grid to a row of 784 numbers
        ↓
Dense(128, activation='relu')   →  finds patterns
        ↓
Dense(128, activation='relu')   →  finds deeper patterns
        ↓
Dense(10, activation='softmax') →  outputs confidence % for each digit (0–9)
```

---

## Key Concepts Explained Simply

- **Epoch** — one full pass through all 60,000 training images.
  Think of it as one complete round of studying.
- **Weight** — a number on every connection between neurons.
  Training = adjusting these numbers until predictions are correct.
- **Normalisation** — dividing pixel values by 255 so they are small (0–1).
  Small numbers make learning faster and more stable.
- **Softmax** — converts raw scores into percentage confidence values.
  Example: `[0.01, 0.02, 0.90, ...]` means 90 % confident the digit is 2.
- **Overfitting** — the model memorises training data but fails on new data.
  Fix: use Dropout layers or reduce epochs.

---

## What Could You Try Next?

- Add a **Dropout layer** (`tf.keras.layers.Dropout(0.2)`) to reduce overfitting.
- Train for **more epochs** (10–20) and see if accuracy improves.
- Try a **Convolutional Neural Network (CNN)** — it reads images like human eyes
  and typically achieves 99 %+ accuracy on MNIST.
- Test the model on a **photo of your own handwriting**
  (resize to 28×28, convert to grayscale, invert colours).

---

## Branch

This lesson was developed on the
`feature/handwritten-digit-classification` branch
and merged into `main`.
