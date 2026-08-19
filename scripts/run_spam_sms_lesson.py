"""Run the full SMS spam detection lesson in a very simple step-by-step order."""

from pathlib import Path

from spam_sms_lesson.pipeline import (
    clean_sms_text,
    compute_text_statistics,
    build_lstm_model,
    encode_labels,
    evaluate_model,
    get_classification_report_text,
    load_sms_dataset,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_wordcloud,
    save_metrics,
    split_data,
    tokenize_and_pad,
    train_model,
)


def locate_sms_csv(project_root: Path) -> Path:
    """Find the SMS CSV file in common locations.

    The script checks a few simple places so beginners do not need to edit code.
    """

    candidates = [
        project_root / "spam_sms_dataset.csv",
        project_root / "sms_spam_dataset.csv",
        project_root / "spam.csv",
        project_root / "spam_ham_dataset.csv",
        project_root / "data" / "raw" / "spam_sms_dataset.csv",
        project_root / "data" / "raw" / "sms_spam_dataset.csv",
        project_root / "data" / "raw" / "spam.csv",
        project_root / "data" / "raw" / "spam_ham_dataset.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    all_csvs = list(project_root.glob("*.csv")) + list((project_root / "data" / "raw").glob("*.csv"))
    if all_csvs:
        return all_csvs[0]

    raise FileNotFoundError(
        "Could not find any CSV file. Place it in project root or data/raw/."
    )


def main() -> None:
    """Run every lesson step in order and print clear messages."""

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"

    print("Step 1: Load SMS dataset")
    csv_path = locate_sms_csv(project_root)
    raw_data = load_sms_dataset(csv_path)
    print(f"Loaded SMS file: {csv_path}")
    print(f"Loaded rows: {len(raw_data)}")

    print("Step 2: Encode labels")
    encoded_data = encode_labels(raw_data)

    print("Step 3: Plot class distribution")
    plot_class_distribution(
        encoded_data,
        title="SMS Spam Class Distribution",
        output_path=artifacts_dir / "plots" / "sms_01_class_distribution.png",
    )

    print("Step 4: Clean SMS text")
    cleaned_data = clean_sms_text(encoded_data)

    print("Step 5: Compute text statistics")
    stats = compute_text_statistics(cleaned_data)
    print(stats)

    print("Step 6: Split the dataset")
    X_train_text, X_test_text, y_train, y_test = split_data(cleaned_data)
    print(f"Training rows: {len(X_train_text)}")
    print(f"Testing rows: {len(X_test_text)}")

    print("Step 7: Tokenize and pad text")
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad(X_train_text, X_test_text)
    print(f"Train tensor shape: {X_train_pad.shape}")
    print(f"Test tensor shape: {X_test_pad.shape}")

    print("Step 8: Build the model")
    model = build_lstm_model(tokenizer)
    model.summary()

    print("Step 9: Train the model")
    history = train_model(model, X_train_pad, y_train, X_test_pad, y_test)

    print("Step 10: Evaluate the model")
    metrics = evaluate_model(model, X_test_pad, y_test)
    report_text = get_classification_report_text(model, X_test_pad, y_test)
    print(metrics)
    print(report_text)

    print("Step 11: Create plots")
    plot_confusion_matrix(
        model,
        X_test_pad,
        y_test,
        artifacts_dir / "plots" / "sms_02_confusion_matrix.png",
    )
    plot_learning_curves(
        history,
        artifacts_dir / "plots" / "sms_03_learning_curves.png",
    )
    plot_wordcloud(
        cleaned_data,
        label_name="ham",
        title="WordCloud: SMS Ham",
        output_path=artifacts_dir / "plots" / "sms_04_wordcloud_ham.png",
    )
    plot_wordcloud(
        cleaned_data,
        label_name="spam",
        title="WordCloud: SMS Spam",
        output_path=artifacts_dir / "plots" / "sms_05_wordcloud_spam.png",
    )

    print("Step 12: Save model and report")
    model_path = artifacts_dir / "models" / "spam_sms_lstm_from_script.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_metrics(metrics, report_text, artifacts_dir / "reports" / "spam_sms_metrics_from_script.txt")

    print("SMS lesson complete.")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
