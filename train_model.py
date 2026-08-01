"""
train_model.py
--------------
Machine Learning baseline model training script for 'The Shelf'.

This script:
  1. Loads clean Goodreads dataset from `datasets/processed/goodreads_merged.csv`.
  2. Splits dataset into 80/20 train/test sets using stratified sampling.
  3. Preprocesses and vectorizes text using TF-IDF (`TfidfVectorizer`).
  4. Trains a linear text classifier (`LinearSVC` with balanced class weights).
  5. Evaluates model performance (Accuracy, full Classification Report, Confusion Matrix, and 5 lowest F1 categories).
  6. Saves trained pipeline artifact to `output/shelf_classifier_pipeline.joblib`.
  7. Performs sanity check predictions on sample text prompts.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
PROCESSED_DATA_PATH = BASE_DIR / "datasets" / "processed" / "goodreads_merged.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_SAVE_PATH = OUTPUT_DIR / "shelf_classifier_pipeline.joblib"


def load_dataset() -> pd.DataFrame:
    """
    Loads labeled dataset from datasets/processed/goodreads_merged.csv.
    """
    logger.info(f"Loading dataset from {PROCESSED_DATA_PATH}...")
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Merged dataset not found at {PROCESSED_DATA_PATH}. Run data_prep.py first.")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Successfully loaded {len(df)} records across {df['shelf_label'].nunique()} categories.")
    return df


def build_pipeline(y_train: pd.Series) -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline with TF-IDF vectorization and LogisticRegression.
    Dynamically computes balanced class weights and applies a targeted 0.7x multiplier to Anime & Manga.
    """
    logger.info("Building TF-IDF + LogisticRegression pipeline with targeted class weight scaling...")
    unique_classes = np.array(sorted(y_train.unique()))
    balanced_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    custom_class_weights = dict(zip(unique_classes, balanced_weights))

    # Apply 0.7x multiplier specifically to Anime & Manga to optimize precision against Fantasy
    if "Anime & Manga" in custom_class_weights:
        custom_class_weights["Anime & Manga"] *= 0.7

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    classifier = LogisticRegression(
        C=1.0,
        class_weight=custom_class_weights,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])

    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, float]:
    """
    Splits dataset into train/test sets, trains model pipeline, and prints diagnostic evaluation metrics.
    """
    X = df["text"].astype(str)
    y = df["shelf_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    logger.info(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    pipeline = build_pipeline(y_train)

    logger.info("Fitting model pipeline on training set...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model on test set...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("               MODEL DIAGNOSTIC EVALUATION")
    print("=" * 60)
    print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")

    print("FULL CLASSIFICATION REPORT:")
    report_text = classification_report(y_test, y_pred, zero_division=0)
    print(report_text)

    # Calculate and print 5 lowest F1-score categories
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    category_f1s = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict) and "f1-score" in metrics and label not in ["accuracy", "macro avg", "weighted avg"]:
            category_f1s.append((label, metrics["f1-score"], metrics["support"]))

    category_f1s.sort(key=lambda x: x[1])
    lowest_5 = category_f1s[:5]

    print("\n" + "-" * 60)
    print("5 LOWEST F1-SCORE SHELF CATEGORIES (Diagnostic Warning):")
    print("-" * 60)
    for rank, (label, f1, support) in enumerate(lowest_5, start=1):
        print(f"  {rank}. {label:<35} | F1-Score: {f1:.4f} | Test Support: {support}")
    print("-" * 60 + "\n")

    print("FULL CONFUSION MATRIX:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())
    print("\n" + "=" * 60 + "\n")

    return pipeline, accuracy


def save_model(pipeline: Pipeline, save_path: Path) -> None:
    """
    Saves trained pipeline artifact to disk using joblib.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Successfully saved trained model pipeline to {save_path}")


def run_sanity_checks(pipeline: Pipeline) -> None:
    """
    Runs sample inference predictions on synthetic text inputs.
    """
    # Sanity check only, not a substitute for the confusion matrix
    sample_prompts = [
        "A dark wizard threatens the magical kingdom with ancient dark spells and dragons.",
        "The history of World War II and European political alliances in the 20th century.",
        "A detective investigates a mysterious murder mystery in a foggy coastal town.",
        "A collection of romantic poems expressing deep love, heartbreak, and emotional devotion.",
        "Deep philosophical thoughts on human consciousness, morality, and the nature of reality.",
        "একটি ভয়ংকর পুরনো রাজবাড়িতে অশরীরী ভূতের রহস্যময় উপদ্রব ও আতঙ্কের গল্প...",
        "দুটি তরুণ হৃদয়ের গভীর ভালোবাসা, প্রেম, ব্যাকুলতা ও আবেগঘন বিরহের অনুভূতি...",
        "ভবিষ্যতের মহাকাশ অভিযান, ভিনগ্রহের প্রাণী ও রোবটের বৈজ্ঞানিক কল্পকাহিনী..."
    ]

    print("SANITY CHECK INFERENCE PREDICTIONS:")
    print("# Sanity check only, not a substitute for the confusion matrix")
    print("-" * 60)
    predictions = pipeline.predict(sample_prompts)
    for prompt, pred in zip(sample_prompts, predictions):
        print(f"Input: \"{prompt[:65]}...\"")
        print(f"Predicted Shelf: [{pred}]\n")
    print("-" * 60 + "\n")


def main() -> None:
    """
    Main training execution function.
    """
    logger.info("Starting 'The Shelf' baseline model training pipeline...")

    # 1. Load clean dataset
    df = load_dataset()

    # 2. Train & Evaluate
    pipeline, accuracy = train_and_evaluate(df)

    # 3. Export fitted pipeline artifact
    save_model(pipeline, MODEL_SAVE_PATH)

    # 4. Sanity check predictions
    run_sanity_checks(pipeline)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
