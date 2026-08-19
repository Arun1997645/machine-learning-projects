from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass(frozen=True)
class NaiveBayesLessonConfig:
    """Simple settings used by the Naive Bayes lesson.

    Keeping settings in one place makes the lesson easier for beginners to edit.
    """

    random_state: int = 42
    test_size: float = 0.2


CONFIG = NaiveBayesLessonConfig()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Step 1: Load the dataset from a CSV file.

    The lesson expects two columns:
    - text: the sentence or short document
    - label: the category name
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    data = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    return data.copy()


def split_features_and_target(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Step 2: Separate text inputs and labels.

    X contains the text.
    y contains the label we want to predict.
    """

    X = data["text"].astype(str)
    y = data["label"].astype(str)
    return X, y


def split_train_test(X: pd.Series, y: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Step 3: Split the dataset into training and testing parts."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train: pd.Series, X_test: pd.Series) -> tuple[CountVectorizer, np.ndarray, np.ndarray]:
    """Step 4: Convert raw text into word-count features.

    CountVectorizer counts how many times each word appears.
    This turns text into numbers that the model can learn from.
    """

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    return vectorizer, X_train_vectorized, X_test_vectorized


def train_naive_bayes_model(X_train_vectorized: np.ndarray, y_train: pd.Series) -> MultinomialNB:
    """Step 5: Train a Multinomial Naive Bayes classifier.

    This model is a classic beginner-friendly algorithm for text classification.
    """

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    return model


def make_predictions(model: MultinomialNB, X_test_vectorized: np.ndarray) -> np.ndarray:
    """Step 6: Predict labels for the test data."""

    return model.predict(X_test_vectorized)


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> tuple[float, np.ndarray, str]:
    """Step 7: Evaluate the model.

    Returns:
    - accuracy
    - confusion matrix
    - classification report text
    """

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    report = classification_report(y_test, y_pred, zero_division=0)
    return float(accuracy), matrix, report


def plot_confusion_matrix(matrix: np.ndarray, class_labels: np.ndarray, output_path: Path) -> None:
    """Plot the confusion matrix heatmap and save it."""

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def predict_single_text(model: MultinomialNB, vectorizer: CountVectorizer, user_input: str) -> str:
    """Step 8: Predict the category of one new text sentence."""

    user_input_vectorized = vectorizer.transform([user_input])
    predicted_label = model.predict(user_input_vectorized)
    return str(predicted_label[0])


def save_metrics_report(accuracy: float, report: str, output_path: Path) -> None:
    """Save evaluation results so the learner can inspect them later."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"accuracy: {accuracy:.4f}\n\nclassification_report:\n{report}",
        encoding="utf-8",
    )
