"""Classify one or more test failure messages with a trained model."""

import argparse
from pathlib import Path

import joblib


def predict(model, messages: list[str]) -> list[dict]:
    probabilities = model.predict_proba(messages)
    classes = list(model.classes_)
    results = []
    for message, row in zip(messages, probabilities):
        best = int(row.argmax())
        results.append(
            {
                "message": message,
                "label": classes[best],
                "confidence": round(float(row[best]), 3),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", nargs="+", help="failure message(s) to classify")
    parser.add_argument("--model", type=Path, default=Path("model/model.joblib"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Run: python train.py")

    model = joblib.load(args.model)
    for result in predict(model, args.messages):
        print(f"{result['label']:<18} ({result['confidence']:.0%})  {result['message']}")


if __name__ == "__main__":
    main()
