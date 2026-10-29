# Test Failure Triage

A small machine learning project that reads an automated test failure message and predicts what kind of problem caused it:

| Label | Meaning |
|---|---|
| `product_bug` | The application under test is wrong |
| `flaky_test` | The test is unreliable (timing, stale elements, race conditions) |
| `environment_issue` | The infrastructure is at fault (network, disk, services) |

The idea comes from daily QA work: when a nightly run produces dozens of failures, the first job is sorting real defects from noise. A model that does a first pass saves that time.

## How it works

1. `generate_data.py` creates a labelled dataset of failure messages from templates. Messages overlap across classes on purpose (a `TimeoutError` can be flaky or environmental) and 4% of labels are flipped to mimic human error.
2. `train.py` turns text into TF-IDF features (word unigrams and bigrams) and trains a logistic regression classifier with scikit-learn. It reports 5-fold cross-validation and held-out test results.
3. `predict.py` loads the saved model and classifies new messages from the command line.
4. `tests/` checks the data generator and the model behavior with pytest.

## Results

On the synthetic dataset (1,500 messages, 80/20 stratified split):

| Metric | Score |
|---|---|
| Cross-validated macro F1 | 0.966 (± 0.017) |
| Held-out accuracy | 0.960 |
| Held-out macro F1 | 0.960 |

Per class on the held-out set:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| environment_issue | 0.95 | 0.96 | 0.95 |
| flaky_test | 0.96 | 0.98 | 0.97 |
| product_bug | 0.97 | 0.94 | 0.95 |

These numbers are high because the data is generated from a limited set of templates. They show the pipeline works. They do not predict performance on real logs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_data.py          # writes data/failures.csv
python train.py                  # trains, prints metrics, saves model/
python predict.py "TimeoutError: element #submit not clickable after 10s, passed on retry"
python -m pytest                 # runs the tests
```

Example output:

```
flaky_test         (91%)  TimeoutError: element #submit not clickable after 10s, passed on retry
```

## Project structure

```
generate_data.py   synthetic dataset generator
train.py           training and evaluation
predict.py         command line predictions
tests/             pytest tests
requirements.txt   dependencies
```

## Limitations

- The data is synthetic. Real failure logs are messier, longer and more varied.
- Only the error message is used. Stack traces, test names and run history would likely help.
- Labels in real projects are subjective, and two engineers may disagree on whether a failure is flaky or a bug.

## Ideas for next steps

- Train on real, anonymised failure logs from a personal or open-source project.
- Compare against a sentence-embedding model and report the difference.
- Add a confusion-matrix plot and per-class error analysis of the mistakes.
- Wrap `predict.py` in a small FastAPI service and add a Dockerfile.
- Add a GitHub Actions workflow that runs the tests on every push.

## Tech

Python, scikit-learn, pandas, joblib, pytest.
