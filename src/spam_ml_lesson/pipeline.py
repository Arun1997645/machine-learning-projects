from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from nltk.corpus import stopwords
from nltk import download as nltk_download
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud


@dataclass(frozen=True)
class LessonConfig:
    """Configuration values used in the teaching pipeline.

    Keeping settings in one place helps beginners change values safely.
    """

    random_state: int = 42
    test_size: float = 0.2
    max_words: int = 12000
    max_len: int = 100
    embedding_dim: int = 32
    lstm_units: int = 16
    dense_units: int = 32
    batch_size: int = 32
    epochs: int = 20


CONFIG = LessonConfig()


def ensure_stopwords_downloaded() -> None:
    """Step 0: Make sure NLTK stopwords are available.

    We use stopwords to remove very common words like "the" and "is"
    so the model can focus on meaningful words.
    """

    nltk_download("stopwords", quiet=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Step 1: Load the CSV file.

    Expected columns: `label` and `text`.
    `label` should contain values like `ham` and `spam`.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    data = pd.read_csv(csv_path)

    required_cols = {"label", "text"}
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    return data.copy()


def plot_class_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Step 2: Plot how many spam and ham samples we have.

    This teaches class imbalance: if one class has far more examples,
    the model can become biased.
    """

    plt.figure(figsize=(7, 4))
    sns.countplot(x="label", data=data)
    plt.title("Original Class Distribution")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def balance_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Balance the dataset by downsampling the majority class.

    If there are more ham messages than spam messages, we randomly pick
    only some ham messages so both classes have the same size.
    """

    ham_df = data[data["label"].str.lower() == "ham"]
    spam_df = data[data["label"].str.lower() == "spam"]

    if ham_df.empty or spam_df.empty:
        raise ValueError("Both `ham` and `spam` classes must exist in the dataset.")

    ham_sample = ham_df.sample(n=len(spam_df), random_state=CONFIG.random_state)
    balanced = (
        pd.concat([ham_sample, spam_df], axis=0)
        .sample(frac=1.0, random_state=CONFIG.random_state)
        .reset_index(drop=True)
    )

    return balanced


def remove_punctuation(text: str) -> str:
    """Step 4a: Remove punctuation characters.

    Example: "hello!!!" becomes "hello".
    """

    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords(text: str) -> str:
    """Step 4b: Remove stopwords from one text sentence.

    This keeps useful words and drops high-frequency helper words.
    """

    stop_words = set(stopwords.words("english"))
    cleaned_words = [word for word in str(text).lower().split() if word not in stop_words]
    return " ".join(cleaned_words)


def clean_text_column(data: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Clean text using multiple small transforms.

    The sequence is:
    1. Lowercase
    2. Remove the word 'subject' (common email prefix)
    3. Remove punctuation
    4. Remove stopwords
    """

    cleaned = data.copy()
    cleaned["text"] = cleaned["text"].astype(str).str.lower().str.replace("subject", "", regex=False)
    cleaned["text"] = cleaned["text"].apply(remove_punctuation)
    cleaned["text"] = cleaned["text"].apply(remove_stopwords)
    return cleaned


def plot_wordcloud(data: pd.DataFrame, label_name: str, title: str, output_path: Path) -> None:
    """Step 5: Draw a word cloud for one class (`ham` or `spam`).

    Word clouds help beginners quickly see the most common words.
    """

    subset = data[data["label"].str.lower() == label_name.lower()]
    corpus = " ".join(subset["text"].astype(str).tolist())

    if not corpus.strip():
        return

    cloud = WordCloud(background_color="black", width=900, height=450, max_words=120).generate(corpus)

    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def tokenize_and_pad(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    """Step 6: Convert text to numbers using tokenization and padding.

    Neural networks need numeric input. We do this in two mini-steps:
    1. Tokenizer maps words to integers.
    2. Padding makes all sequences equal length.
    """

    labels = (data["label"].str.lower() == "spam").astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        labels,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=labels,
    )

    tokenizer = Tokenizer(num_words=CONFIG.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train.tolist())

    train_sequences = tokenizer.texts_to_sequences(X_train.tolist())
    test_sequences = tokenizer.texts_to_sequences(X_test.tolist())

    X_train_pad = pad_sequences(train_sequences, maxlen=CONFIG.max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(test_sequences, maxlen=CONFIG.max_len, padding="post", truncating="post")

    return X_train_pad, X_test_pad, y_train, y_test, tokenizer


def build_model(tokenizer: Tokenizer) -> tf.keras.Model:
    """Step 7: Build a simple LSTM model.

    Architecture:
    1. Embedding layer (turns word IDs into dense vectors)
    2. LSTM layer (learns sequence patterns)
    3. Dense + ReLU (small decision block)
    4. Dense + Sigmoid (binary output: spam or not spam)
    """

    vocab_size = min(CONFIG.max_words, len(tokenizer.word_index) + 1)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=CONFIG.embedding_dim,
                input_length=CONFIG.max_len,
            ),
            tf.keras.layers.LSTM(CONFIG.lstm_units),
            tf.keras.layers.Dense(CONFIG.dense_units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tf.keras.callbacks.History:
    """Step 8: Train model using early stopping and LR scheduling.

    EarlyStopping prevents over-training.
    ReduceLROnPlateau lowers learning rate when progress slows.
    """

    early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

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
    """Step 9: Evaluate model on unseen test data."""

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {"test_loss": float(loss), "test_accuracy": float(accuracy)}


def plot_accuracy_curve(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Step 10: Plot training and validation accuracy curves."""

    plt.figure(figsize=(7, 4))
    plt.plot(history.history.get("accuracy", []), label="Training Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Step 11: Save metrics to a text file for easy review."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
