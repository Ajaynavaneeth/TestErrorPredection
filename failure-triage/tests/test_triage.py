import joblib
import pytest

from generate_data import LABELS, generate
from predict import predict
from train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp("triage")
    csv_path = work / "failures.csv"
    rows = generate(n_samples=900, noise=0.04, seed=7)
    csv_path.write_text(
        "text,label\n" + "\n".join(f'"{text}",{label}' for text, label in rows),
        encoding="utf-8",
    )
    metrics = train(csv_path, work / "model")
    model = joblib.load(work / "model" / "model.joblib")
    return model, metrics


def test_generate_is_reproducible():
    assert generate(50, 0.04, seed=1) == generate(50, 0.04, seed=1)


def test_generate_covers_all_labels():
    labels = {label for _, label in generate(300, 0.0, seed=3)}
    assert labels == set(LABELS)


def test_model_beats_chance_by_a_wide_margin(trained):
    _, metrics = trained
    assert metrics["test_f1_macro"] > 0.8


def test_obvious_environment_failure(trained):
    model, _ = trained
    result = predict(model, ["OSError: No space left on device while writing /tmp/report.xml"])[0]
    assert result["label"] == "environment_issue"


def test_obvious_product_bug(trained):
    model, _ = trained
    result = predict(model, ["NullPointerException at OrderService.applyDiscount(line 88)"])[0]
    assert result["label"] == "product_bug"


def test_confidence_is_a_probability(trained):
    model, _ = trained
    result = predict(model, ["something completely unrelated"])[0]
    assert 0.0 <= result["confidence"] <= 1.0
