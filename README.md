# The Shelf Training Pipeline (`The-Shelf-Training`)

Machine learning dataset preparation, training, evaluation, and parameter export pipeline for **The Shelf**, a cross-platform application designed to organize books, documents, and digital media into auto-categorized shelves using on-device text classification.

---

## Technical Overview

The training pipeline processes document and book metadata (titles, synopsis details, and genre tags) to train an on-device text classifier across 17 target shelf categories:

- **Classifier Architecture**: `TfidfVectorizer` (sublinear TF scaling, 10,000 max features, 1–2 n-grams) combined with `LogisticRegression` (`C=1.0`, `solver='lbfgs'`, `max_iter=1000`).
- **Target Categories (17 Shelves)**: Fantasy, Historical Fiction, Mystery, Romance, Science Fiction, Horror, Thriller, Young Adult, Graphic Novels & Comics, Anime & Manga, Children's, Poetry, History, Biography & Memoir, Philosophy, Self-Help & Personal Development, and Miscellaneous.
- **Data Remediation Engine**: High-precision text pattern regex scanning (`TEXT_FALLBACK_PATTERNS`) in `data_prep.py` for books with generic or missing Goodreads genre tags.
- **Targeted Class Weight Scaling**: Dynamic class weighting via `compute_class_weight('balanced', ...)` with a targeted `0.7x` multiplier applied specifically to `Anime & Manga` in `train_model.py` to eliminate decision-boundary overlap with `Fantasy`.
- **Pure Dart Model Export**: Full model parameter serialization (`output/tfidf_model.json`) in `export_model.py`, enabling bit-exact Pure Dart on-device inference without native C++ dependencies.

---

## Repository Structure

```text
the-shelf-training/
├── data_prep.py             # Genre tag mapping, regex fallback scanning, and dataset merging engine
├── train_model.py           # Model training, dynamic class weighting, and diagnostic evaluation pipeline
├── tune_experiments.py      # Non-destructive hyperparameter tuning and model experiment runner
├── export_model.py          # Model parameter serialization (coef, intercept, vocabulary, idf) script
├── test_shelf_classifier.dart # Standalone Pure Dart inference verification script
├── fetch_api_data.py        # API ingest script for Jikan (Kitsu) and Rokomari dataset enrichment
├── scrape_rokomari.py       # Supplemental web scraping utility for Bengali book metadata
├── requirements.txt         # Python environment dependencies
├── datasets/
│   ├── raw/                 # Raw source CSV data files
│   └── processed/           # Processed primary dataset (goodreads_merged.csv)
└── output/
    ├── shelf_classifier_pipeline.joblib  # Serialized Scikit-Learn pipeline checkpoint
    └── tfidf_model.json                  # Exported JSON model parameters for Flutter client
```

---

## Setup and Installation

### 1. Environment Initialization

Create and activate an isolated Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Dependency Installation

Upgrade `pip` and install all required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Pipeline Execution Workflow

Execute the pipeline scripts sequentially:

### Step 1: Data Preparation and Merging

Clean source data, execute priority-ordered genre tag mapping, apply high-precision text regex fallbacks for missing tags, and output the merged dataset:

```bash
python data_prep.py
```

Outputs the cleaned primary dataset to `datasets/processed/goodreads_merged.csv`.

### Step 2: Production Model Training

Train the production classifier, compute balanced class weights with the `0.7x` `Anime & Manga` multiplier, evaluate metrics, and save the model checkpoint:

```bash
python train_model.py
```

Generates diagnostic reports (Accuracy, Full Classification Report, 17-Class Confusion Matrix) and serializes the pipeline to `output/shelf_classifier_pipeline.joblib`.

### Step 3: Model Parameter Export

Extract weight matrices, bias vectors, vocabulary indices, and IDF values into a single JSON asset:

```bash
python export_model.py
```

Generates `output/tfidf_model.json` (3.85 MB) containing all mathematical parameters required for client-side vectorization and classification.

### Step 4: Standalone Pure Dart Verification

Verify parameter correctness and probability equivalence against Scikit-Learn using the standalone Dart test runner:

```bash
dart test_shelf_classifier.dart
```

Evaluates English and Bengali test prompts, confirming bit-exact probability identity ($0.000000$ diff) and sub-millisecond execution latency.

---

## Performance Summary

- **Overall Accuracy**: 58.08%
- **Macro F1 Score**: 0.4926
- **Anime & Manga Precision**: 0.6721
- **Anime & Manga F1 Score**: 0.6694
- **Fantasy F1 Score**: 0.7290
- **Pure Dart Inference Latency**: < 0.1 ms per sentence
