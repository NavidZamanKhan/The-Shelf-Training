"""
train_model.py
--------------
Machine Learning model training script for 'The Shelf'.

This script:
  1. Loads processed datasets from `datasets/`.
  2. Extracts feature text (title + description) and target shelf categories.
  3. Preprocesses and vectorizes text using TF-IDF (`TfidfVectorizer`).
  4. Trains a text classifier (Linear SVM / Naive Bayes).
  5. Evaluates model performance (Accuracy, Classification Report).
  6. Saves trained model artifacts into `output/`.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_SAVE_PATH = OUTPUT_DIR / "shelf_classifier_pipeline.joblib"


def load_datasets() -> pd.DataFrame:
    """
    Loads and merges all JSON/CSV datasets from the `datasets/` directory.
    Falls back to a synthetic dataset if no collected data is available yet.
    
    Returns:
        pd.DataFrame: DataFrame containing 'text' and 'shelf_label' columns.
    """
    logger.info("Loading training datasets...")
    records: List[Dict[str, Any]] = []

    # Read all JSON files in datasets/
    json_files = list(DATASETS_DIR.glob("*.json"))
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
            logger.info(f"Loaded {len(data)} records from {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")

    if records:
        df = pd.DataFrame(records)
        # Construct unified feature text
        if "description" in df.columns and "title" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        elif "description" in df.columns:
            df["text"] = df["description"]
        else:
            df["text"] = df["title"]

        # Ensure label column exists
        if "shelf_label" not in df.columns and "genre" in df.columns:
            df["shelf_label"] = df["genre"]
            
        return df[["text", "shelf_label"]].dropna()

    # Fallback synthetic placeholder dataset for initial testing
    logger.warning("No dataset files found in datasets/. Using synthetic placeholder dataset.")
    placeholder_data = [
        {"text": "Introduction to Quantum Physics and Thermodynamics", "shelf_label": "Science"},
        {"text": "Astrophysics for People in a Hurry Cosmology Space", "shelf_label": "Science"},
        {"text": "Clean Code Principles of Agile Software Craftsmanship Python Java", "shelf_label": "Technology"},
        {"text": "Designing Data Intensive Applications Distributed Systems", "shelf_label": "Technology"},
        {"text": "The Rise and Fall of the Third Reich World War History", "shelf_label": "History"},
        {"text": "Sapiens A Brief History of Humankind Civilizations", "shelf_label": "History"},
        {"text": "Dune Frank Herbert Epic Sci-Fi Planet Arrakis Spice", "shelf_label": "Sci-Fi"},
        {"text": "Foundation Isaac Asimov Galactic Empire Psychohistory", "shelf_label": "Sci-Fi"},
    ] * 5  # Duplicate to provide enough samples for train/test split
    
    return pd.DataFrame(placeholder_data)


def build_pipeline(classifier_type: str = "linear_svm") -> Pipeline:
    """
    Constructs an end-to-end Scikit-Learn Pipeline with TF-IDF vectorization and classifier.
    
    Args:
        classifier_type (str): 'linear_svm' (LinearSVC) or 'naive_bayes' (MultinomialNB).
        
    Returns:
        Pipeline: Scikit-learn Pipeline object.
    """
    logger.info(f"Building pipeline with TF-IDF + {classifier_type}...")
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    if classifier_type == "linear_svm":
        classifier = LinearSVC(C=1.0, random_state=42)
    elif classifier_type == "naive_bayes":
        classifier = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier)
    ])
    
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, float]:
    """
    Splits dataset into train/test sets, trains model pipeline, and logs evaluation metrics.
    
    Args:
        df (pd.DataFrame): Input dataset with 'text' and 'shelf_label' columns.
        
    Returns:
        Tuple[Pipeline, float]: Trained pipeline and test accuracy.
    """
    X = df["text"]
    y = df["shelf_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    logger.info(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples.")

    pipeline = build_pipeline(classifier_type="linear_svm")
    
    logger.info("Fitting model pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, zero_division=0))
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    return pipeline, accuracy


def save_model(pipeline: Pipeline, save_path: Path) -> None:
    """
    Saves trained pipeline artifact to disk using joblib.
    
    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        save_path (Path): Destination file path.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Successfully saved trained model pipeline to {save_path}")


def main() -> None:
    """
    Main training execution function.
    """
    logger.info("Starting 'The Shelf' model training pipeline...")
    
    # 1. Load data
    df = load_datasets()
    logger.info(f"Total dataset size: {len(df)} samples across {df['shelf_label'].nunique()} shelf categories.")

    # 2. Train & Evaluate
    pipeline, accuracy = train_and_evaluate(df)

    # 3. Export fitted pipeline artifact
    save_model(pipeline, MODEL_SAVE_PATH)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
