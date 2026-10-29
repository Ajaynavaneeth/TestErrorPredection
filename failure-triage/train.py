"""Train and evaluate a classifier that triages test failure messages."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train(csv_path: Path, model_dir: Path) -> dict:
    data = pd.read_csv(csv_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.2,
        stratify=data["label"],
        random_state=RANDOM_STATE,
    )

    model = build_pipeline()
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="f1_macro")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    labels = sorted(data["label"].unique())
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    metrics = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "test_accuracy": round(float(report["accuracy"]), 4),
        "test_f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in labels
        },
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/failures.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    args = parser.parse_args()

    metrics = train(args.data, args.model_dir)
    print(f"Cross-validated macro F1: {metrics['cv_f1_macro_mean']} (+/- {metrics['cv_f1_macro_std']})")
    print(f"Held-out test accuracy:   {metrics['test_accuracy']}")
    print(f"Held-out test macro F1:   {metrics['test_f1_macro']}\n")
    print("Per class:")
    for label, scores in metrics["per_class"].items():
        print(f"  {label:<18} precision {scores['precision']:.2f}  recall {scores['recall']:.2f}  f1 {scores['f1']:.2f}")
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print("  labels:", metrics["labels"])
    for row in metrics["confusion_matrix"]:
        print(" ", row)
    print(f"\nSaved model and metrics to {args.model_dir}/")


if __name__ == "__main__":
    main()
