"""Run the full spam detection lesson, step by step.

This script is intentionally beginner-friendly.
Each printed section maps to one learning step.
"""

from pathlib import Path

from spam_ml_lesson.pipeline import (
    balance_dataset,
    build_model,
    clean_text_column,
    ensure_stopwords_downloaded,
    evaluate_model,
    load_dataset,
    plot_accuracy_curve,
    plot_class_distribution,
    plot_wordcloud,
    save_metrics,
    tokenize_and_pad,
    train_model,
)


def locate_csv(project_root: Path) -> Path:
    """Find the CSV file in common beginner locations.

    Priority:
    1. Project root
    2. data/raw
    """

    candidates = [
        project_root / "spam_ham_dataset.csv",
        project_root / "data" / "raw" / "spam_ham_dataset.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find spam_ham_dataset.csv. Place it in project root or data/raw/."
    )


def main() -> None:
    """Execute all lesson steps in a clear sequence."""

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"

    print("Step 0: Prepare stopwords")
    ensure_stopwords_downloaded()

    print("Step 1: Load dataset")
    csv_path = locate_csv(project_root)
    data = load_dataset(csv_path)
    print(f"Loaded rows: {len(data)}")

    print("Step 2: Plot original class distribution")
    plot_class_distribution(data, artifacts_dir / "plots" / "01_original_distribution.png")

    print("Step 3: Balance the dataset")
    balanced = balance_dataset(data)
    plot_class_distribution(balanced, artifacts_dir / "plots" / "02_balanced_distribution.png")

    print("Step 4: Clean text")
    cleaned = clean_text_column(balanced)

    print("Step 5: Build word clouds")
    plot_wordcloud(
        cleaned,
        label_name="ham",
        title="WordCloud for Ham Emails",
        output_path=artifacts_dir / "plots" / "03_wordcloud_ham.png",
    )
    plot_wordcloud(
        cleaned,
        label_name="spam",
        title="WordCloud for Spam Emails",
        output_path=artifacts_dir / "plots" / "04_wordcloud_spam.png",
    )

    print("Step 6: Tokenize and pad")
    X_train, X_test, y_train, y_test, tokenizer = tokenize_and_pad(cleaned)

    print("Step 7: Build model")
    model = build_model(tokenizer)
    model.summary()

    print("Step 8: Train model")
    history = train_model(model, X_train, y_train, X_test, y_test)

    print("Step 9: Evaluate model")
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

    print("Step 10: Plot accuracy curve")
    plot_accuracy_curve(history, artifacts_dir / "plots" / "05_accuracy_curve.png")

    print("Step 11: Save model and metrics")
    model_save_path = artifacts_dir / "models" / "spam_lstm.keras"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    save_metrics(metrics, artifacts_dir / "reports" / "metrics.txt")

    print("Lesson complete.")
    print(f"Model saved at: {model_save_path}")


if __name__ == "__main__":
    main()
