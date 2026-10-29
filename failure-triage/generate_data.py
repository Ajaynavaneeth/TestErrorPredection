"""Generate a synthetic dataset of automated-test failure messages.

Each message is labelled with the kind of problem behind it:

    product_bug        the application under test is wrong
    flaky_test         the test is unreliable (timing, stale elements, races)
    environment_issue  the infrastructure is the problem (network, disk, services)

The messages overlap on purpose. For example, a TimeoutError can be a flaky
test or an environment problem, and a few labels are flipped to mimic human
mistakes. That keeps the task realistic instead of trivially separable.

Replace this file's output with your own anonymised failure logs when you
have them. Never commit logs that belong to an employer or client.
"""

import argparse
import csv
import random
from pathlib import Path

LABELS = ["product_bug", "flaky_test", "environment_issue"]

TEMPLATES = {
    "product_bug": [
        "AssertionError: expected {exp} but got {got} in {test}",
        "AssertionError: {field} should be {exp}, found {got}",
        "NullPointerException at {cls}.{method}(line {n})",
        "KeyError: '{field}' missing from response body for {endpoint}",
        "HTTP 500 Internal Server Error returned by {endpoint}",
        "ValueError: invalid value for {field} in {test}",
        "TypeError: unsupported operand while computing {field} in {cls}.{method}",
        "Expected status 200 but received 400 from {endpoint}",
    ],
    "flaky_test": [
        "TimeoutError: element {selector} not clickable after {n}s, passed on retry",
        "StaleElementReferenceException: element {selector} is no longer attached to the page",
        "AssertionError: expected {exp} but got {got}, passed on rerun",
        "Intermittent failure in {test}: race condition waiting for {selector}",
        "TimeoutError waiting for {endpoint} response after {n}s",
        "ElementClickInterceptedException: {selector} obscured by overlay, second attempt passed",
        "Test {test} failed 1 of 5 runs with no code change",
        "NoSuchElementException: {selector} not found yet, animation still running",
    ],
    "environment_issue": [
        "ConnectionRefusedError: could not connect to {host}:{port}",
        "TimeoutError: connection to {host} timed out after {n}s",
        "OSError: No space left on device while writing {path}",
        "WebDriverException: chrome not reachable on {host}",
        "DNS resolution failed for {host}",
        "HTTP 503 Service Unavailable from {host}",
        "PermissionError: cannot read {path} on build agent",
        "Database connection pool exhausted on {host} after {n} attempts",
    ],
}

FIELDS = ["email", "total", "status", "user_id", "currency", "quantity", "discount"]
ENDPOINTS = ["/api/orders", "/api/login", "/api/cart", "/api/profile", "/api/payments"]
SELECTORS = ["#submit", "#checkout-btn", ".nav-menu", "#search-box", ".modal-close"]
HOSTS = ["db-01", "selenium-grid", "staging-api", "ci-runner-3", "redis-cache"]
CLASSES = ["OrderService", "CartManager", "UserController", "PriceCalculator"]
METHODS = ["calculateTotal", "applyDiscount", "validateUser", "buildResponse"]
TESTS = ["test_checkout_flow", "test_login", "test_search_results", "test_apply_coupon"]
PATHS = ["/tmp/report.xml", "/var/log/app.log", "/builds/artifacts/out.zip"]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        exp=rng.choice(["'ok'", "42", "'active'", "100.00", "True"]),
        got=rng.choice(["'error'", "0", "None", "99.99", "False"]),
        field=rng.choice(FIELDS),
        endpoint=rng.choice(ENDPOINTS),
        selector=rng.choice(SELECTORS),
        host=rng.choice(HOSTS),
        port=rng.choice([5432, 6379, 4444, 8080]),
        cls=rng.choice(CLASSES),
        method=rng.choice(METHODS),
        test=rng.choice(TESTS),
        path=rng.choice(PATHS),
        n=rng.randint(1, 120),
    )


def generate(n_samples: int, noise: float, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        label = rng.choice(LABELS)
        text = fill(rng.choice(TEMPLATES[label]), rng)
        if rng.random() < noise:
            label = rng.choice([other for other in LABELS if other != label])
        rows.append((text, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=0.04, help="share of flipped labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/failures.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.samples, args.noise, args.seed)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
