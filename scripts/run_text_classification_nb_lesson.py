"""Run the full text classification lesson using Multinomial Naive Bayes."""

from pathlib import Path

import numpy as np

from text_classification_nb_lesson.pipeline import (
    evaluate_model,
    load_dataset,
    make_predictions,
    plot_confusion_matrix,
    predict_single_text,
    save_metrics_report,
    split_features_and_target,
    split_train_test,
    train_naive_bayes_model,
    vectorize_text,
)


def locate_dataset(project_root: Path) -> Path:
    """Find the synthetic text classification dataset in common locations."""

    candidates = [
        project_root / "synthetic_text_data.csv",
        project_root / "data" / "raw" / "synthetic_text_data.csv",
        project_root / "data" / "processed" / "synthetic_text_data.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find synthetic_text_data.csv in expected locations.")


def main() -> None:
    """Run each teaching step in a clear order."""

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"

    print("Step 1: Load dataset")
    csv_path = locate_dataset(project_root)
    data = load_dataset(csv_path)
    print(f"Loaded file: {csv_path}")
    print(f"Shape: {data.shape}")

    print("Step 2: Split text and labels")
    X, y = split_features_and_target(data)
    print(f"Number of text rows: {len(X)}")
    print(f"Labels: {sorted(y.unique())}")

    print("Step 3: Train/test split")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print("Step 4: Convert text into word-count features")
    vectorizer, X_train_vectorized, X_test_vectorized = vectorize_text(X_train, X_test)
    print(f"Training matrix shape: {X_train_vectorized.shape}")
    print(f"Testing matrix shape: {X_test_vectorized.shape}")

    print("Step 5: Train Naive Bayes model")
    model = train_naive_bayes_model(X_train_vectorized, y_train)
    print("Model training completed.")

    print("Step 6: Make predictions")
    y_pred = make_predictions(model, X_test_vectorized)
    print(f"First few predictions: {y_pred[:5]}")

    print("Step 7: Evaluate the model")
    accuracy, matrix, report = evaluate_model(y_test, y_pred)
    class_labels = np.unique(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(report)
    plot_confusion_matrix(
        matrix,
        class_labels,
        artifacts_dir / "plots" / "text_nb_confusion_matrix.png",
    )
    save_metrics_report(
        accuracy,
        report,
        artifacts_dir / "reports" / "text_nb_metrics_from_script.txt",
    )

    print("Step 8: Predict one new text sentence")
    user_input = "I love artificial intelligence and machine learning"
    predicted_label = predict_single_text(model, vectorizer, user_input)
    print(f"The input text belongs to the '{predicted_label}' category.")

    print("Naive Bayes text classification lesson complete.")


if __name__ == "__main__":
    main()
