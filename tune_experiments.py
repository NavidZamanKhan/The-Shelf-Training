"""
tune_experiments.py
-------------------
Hyperparameter & Model Tuning Experiments for 'The Shelf' classification task.

Evaluates 6 variants on `datasets/processed/goodreads_merged.csv` using an 80/20 train/test split:
  - Variant 0: Baseline (LinearSVC C=1.0, max_features=10000, ngram=(1,2))
  - Variant 1: Regularization (LinearSVC C=0.2, max_features=10000, ngram=(1,2))
  - Variant 2: Trigrams (LinearSVC C=1.0, max_features=10000, ngram=(1,3))
  - Variant 3: Reduced Vocabulary (LinearSVC C=1.0, max_features=5000, ngram=(1,2))
  - Variant 4: Naive Bayes (MultinomialNB alpha=0.1, max_features=10000, ngram=(1,2))
  - Variant 5: Logistic Regression (LogReg C=1.0, balanced, max_features=10000, ngram=(1,2))

Outputs results to console and writes CSV to `output/experiments/tuning_summary.csv`.
Does NOT alter `train_model.py` or `output/shelf_classifier_pipeline.joblib`.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DATA_PATH = BASE_DIR / "datasets" / "processed" / "goodreads_merged.csv"
EXPERIMENTS_OUTPUT_DIR = BASE_DIR / "output" / "experiments"
CSV_SAVE_PATH = EXPERIMENTS_OUTPUT_DIR / "tuning_summary.csv"

# Target weak categories to track specifically
TARGET_CATEGORIES = [
    "Thriller",
    "Poetry",
    "Philosophy",
    "Graphic Novels",
    "Humor",
    "Miscellaneous"
]


def load_dataset() -> pd.DataFrame:
    """Loads clean dataset from datasets/processed/goodreads_merged.csv."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found at {PROCESSED_DATA_PATH}.")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Loaded dataset: {len(df)} rows across {df['shelf_label'].nunique()} categories.")
    return df


def get_variants() -> List[Dict[str, Any]]:
    """Defines the 6 experiment configuration variants."""
    return [
        {
            "variant_id": "Var 0 (Baseline)",
            "description": "LinearSVC (C=1.0, 10k feat, 1-2 ngrams)",
            "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=42)
        },
        {
            "variant_id": "Var 1",
            "description": "LinearSVC (C=0.2, 10k feat, 1-2 ngrams)",
            "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            "classifier": LinearSVC(C=0.2, class_weight="balanced", random_state=42)
        },
        {
            "variant_id": "Var 2",
            "description": "LinearSVC (C=1.0, 10k feat, 1-3 ngrams)",
            "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english", sublinear_tf=True),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=42)
        },
        {
            "variant_id": "Var 3",
            "description": "LinearSVC (C=1.0, 5k feat, 1-2 ngrams)",
            "vectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            "classifier": LinearSVC(C=1.0, class_weight="balanced", random_state=42)
        },
        {
            "variant_id": "Var 4",
            "description": "MultinomialNB (alpha=0.1, 10k feat, 1-2 ngrams)",
            "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            "classifier": MultinomialNB(alpha=0.1)
        },
        {
            "variant_id": "Var 5",
            "description": "LogisticRegression (C=1.0, balanced, 10k feat)",
            "vectorizer": TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
            "classifier": LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=42)
        }
    ]


def run_experiments():
    df = load_dataset()
    X = df["text"].astype(str)
    y = df["shelf_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples.")

    variants = get_variants()
    results = []
    baseline_acc = None

    for var in variants:
        vid = var["variant_id"]
        desc = var["description"]
        logger.info(f"Running {vid}: {desc}...")

        pipeline = Pipeline([
            ("tfidf", var["vectorizer"]),
            ("classifier", var["classifier"])
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        if baseline_acc is None:
            baseline_acc = acc
        delta_acc = acc - baseline_acc

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        macro_f1 = report["macro avg"]["f1-score"]

        # Extract per-target-category F1s
        cat_f1s = {}
        for cat in TARGET_CATEGORIES:
            cat_f1s[cat] = report.get(cat, {}).get("f1-score", 0.0)

        row = {
            "Variant ID": vid,
            "Description": desc,
            "Overall Acc (%)": f"{acc * 100:.2f}%",
            "Delta Acc (%)": f"{delta_acc * 100:+.2f}%",
            "Macro F1": round(macro_f1, 4),
            "Thriller F1": round(cat_f1s["Thriller"], 4),
            "Poetry F1": round(cat_f1s["Poetry"], 4),
            "Philosophy F1": round(cat_f1s["Philosophy"], 4),
            "Graphic Novels F1": round(cat_f1s["Graphic Novels"], 4),
            "Humor F1": round(cat_f1s["Humor"], 4),
            "Miscellaneous F1": round(cat_f1s["Miscellaneous"], 4)
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    # Save CSV
    EXPERIMENTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_SAVE_PATH, index=False)
    logger.info(f"Saved experiment results summary CSV to {CSV_SAVE_PATH}")

    # Print nicely formatted table to console
    print("\n" + "=" * 115)
    print("                                MODEL & HYPERPARAMETER TUNING EXPERIMENTS")
    print("=" * 115)
    print(results_df.to_string(index=False))
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_experiments()
