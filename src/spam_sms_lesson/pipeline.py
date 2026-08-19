from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud


@dataclass(frozen=True)
class SmsLessonConfig:
    """Simple configuration values used by the SMS lesson.

    Keeping all important numbers here makes the code easier for beginners to edit.
    """

    random_state: int = 42
    test_size: float = 0.2
    max_words: int = 12000
    max_len: int = 25
    embedding_dim: int = 32
    lstm_units: int = 16
    dense_units: int = 16
    batch_size: int = 32
    epochs: int = 10


CONFIG = SmsLessonConfig()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def load_sms_dataset(csv_path: Path) -> pd.DataFrame:
    """Step 1: Load the SMS CSV file.

    This function supports two common beginner formats:
    1. Columns already named `label` and `text`
    2. The raw SMS spam dataset format using `v1` and `v2`
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    data = pd.read_csv(csv_path, encoding="latin-1")

    if {"label", "text"}.issubset(data.columns):
        cleaned = data[["label", "text"]].copy()
    elif {"v1", "v2"}.issubset(data.columns):
        cleaned = data.copy()
        drop_columns = [column for column in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"] if column in cleaned.columns]
        if drop_columns:
            cleaned = cleaned.drop(columns=drop_columns)
        cleaned = cleaned.rename(columns={"v1": "label", "v2": "text"})
        cleaned = cleaned[["label", "text"]].copy()
    else:
        raise ValueError(
            "CSV must contain either columns ['label', 'text'] or ['v1', 'v2']."
        )

    cleaned = cleaned.dropna(subset=["label", "text"]).reset_index(drop=True)
    cleaned["label"] = cleaned["label"].astype(str).str.strip().str.lower()
    cleaned["text"] = cleaned["text"].astype(str)

    valid_labels = {"ham", "spam"}
    found_labels = set(cleaned["label"].unique())
    if not found_labels.issubset(valid_labels):
        raise ValueError(f"Labels must be only 'ham' or 'spam'. Found: {sorted(found_labels)}")

    return cleaned


def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Add a numeric label column.

    ham becomes 0.
    spam becomes 1.
    """

    encoded = data.copy()
    encoded["label_enc"] = encoded["label"].map({"ham": 0, "spam": 1})
    return encoded


def plot_class_distribution(data: pd.DataFrame, title: str, output_path: Path) -> None:
    """Step 3: Plot how many ham and spam messages we have."""

    plt.figure(figsize=(7, 4))
    sns.countplot(x="label", data=data)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def remove_punctuation(text: str) -> str:
    """Remove punctuation from one message."""

    table = str.maketrans("", "", string.punctuation)
    return str(text).translate(table)


def remove_stop_words(text: str) -> str:
    """Remove very common words that usually do not help classification."""

    words = str(text).lower().split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)


def clean_sms_text(data: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Clean the SMS messages in very small, easy steps.

    We do four things:
    1. convert text to lowercase
    2. remove punctuation
    3. remove stop words
    4. keep the final clean text
    """

    cleaned = data.copy()
    cleaned["text"] = cleaned["text"].astype(str).str.lower()
    cleaned["text"] = cleaned["text"].apply(remove_punctuation)
    cleaned["text"] = cleaned["text"].apply(remove_stop_words)
    return cleaned


def compute_text_statistics(data: pd.DataFrame) -> dict[str, int]:
    """Step 5: Compute simple SMS text statistics.

    This helps beginners understand the data before modeling.
    """

    word_counts = data["text"].astype(str).apply(lambda value: len(value.split()))
    average_words = int(round(word_counts.mean()))
    vocabulary_size = len(set(" ".join(data["text"].astype(str)).split()))

    return {
        "total_rows": int(len(data)),
        "average_words_per_message": average_words,
        "approx_vocabulary_size": int(vocabulary_size),
    }


def split_data(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Step 6: Split messages into training and testing parts."""

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label_enc"].to_numpy(),
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=data["label_enc"],
    )
    return X_train, X_test, y_train, y_test


def tokenize_and_pad(X_train: pd.Series, X_test: pd.Series) -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Step 7: Convert SMS text into padded number sequences."""

    tokenizer = Tokenizer(num_words=CONFIG.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train.tolist())

    X_train_sequences = tokenizer.texts_to_sequences(X_train.tolist())
    X_test_sequences = tokenizer.texts_to_sequences(X_test.tolist())

    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=CONFIG.max_len,
        padding="post",
        truncating="post",
    )
    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=CONFIG.max_len,
        padding="post",
        truncating="post",
    )

    return X_train_padded, X_test_padded, tokenizer


def build_lstm_model(tokenizer: Tokenizer) -> tf.keras.Model:
    """Step 8: Build one simple LSTM model for SMS spam detection.

    The model is intentionally small and easy to understand.
    """

    vocab_size = min(CONFIG.max_words, len(tokenizer.word_index) + 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=CONFIG.embedding_dim),
        tf.keras.layers.LSTM(CONFIG.lstm_units),
        tf.keras.layers.Dense(CONFIG.dense_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.build((None, CONFIG.max_len))
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tf.keras.callbacks.History:
    """Step 9: Train the model using simple safety callbacks."""

    early_stop = EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=CONFIG.epochs,
        batch_size=CONFIG.batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Step 10: Evaluate the final model with common classification metrics."""

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }


def get_classification_report_text(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """Return the classification report as plain text for easy printing and saving."""

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    return classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0)


def plot_confusion_matrix(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, output_path: Path) -> None:
    """Step 11: Plot the confusion matrix."""

    probabilities = model.predict(X_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("SMS Spam Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_learning_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Step 12: Plot training and validation accuracy and loss."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history.get("loss", []), label="Train Loss")
    axes[1].plot(history.history.get("val_loss", []), label="Val Loss")
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_wordcloud(data: pd.DataFrame, label_name: str, title: str, output_path: Path) -> None:
    """Create a word cloud for ham or spam SMS messages."""

    text_blob = " ".join(data[data["label"] == label_name]["text"].astype(str).tolist())
    if not text_blob.strip():
        return

    cloud = WordCloud(background_color="black", width=900, height=450, max_words=120).generate(text_blob)
    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics: dict[str, float], report_text: str, output_path: Path) -> None:
    """Save metrics and classification report into one simple text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"test_loss: {metrics['test_loss']:.4f}",
        f"test_accuracy: {metrics['test_accuracy']:.4f}",
        f"precision: {metrics['precision']:.4f}",
        f"recall: {metrics['recall']:.4f}",
        f"f1_score: {metrics['f1_score']:.4f}",
        "",
        "classification_report:",
        report_text,
    ]
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
