# The Shelf Training Pipeline (`The-Shelf-Training`)

Machine learning training and export pipeline for **The Shelf**, a mobile application designed to organize books, documents, and digital media into auto-categorized shelves using on-device text classification.

---

## Technical Overview

The pipeline processes book metadata (titles and descriptions) to classify items across 16 target shelf categories:

- **Classifier Architecture**: `TfidfVectorizer` (sublinear TF scaling, max 10,000 features, 1–2 n-grams) coupled with `LogisticRegression` (`C=1.0`, `class_weight='balanced'`, `solver='lbfgs'`).
- **Category Mapping**: 16 target shelf categories following the strategic integration of closely adjacent genre tags (such as merging _Thriller_ into _Mystery_).
- **On-Device Target**: Scikit-learn pipeline serialization via `joblib`, JSON metadata export (`tfidf_vocab.json`), and ONNX model conversion (`shelf_classifier.onnx`) for cross-platform Flutter mobile deployment.

---

## Repository Structure

```text
the-shelf-training/
├── data_prep.py           # Label mapping, genre parsing, and dataset merging engine
├── train_model.py         # Production model training and diagnostic evaluation pipeline
├── tune_experiments.py    # Non-destructive hyperparameter and model experiment runner
├── export_model.py        # Model serialization, vocabulary extraction, and ONNX export script
├── fetch_api_data.py      # Public API data ingest script (Open Library, Jikan / MyAnimeList)
├── scrape_data.py         # Web scraping utility (Playwright + stealth) for supplemental data
├── requirements.txt       # Python environment dependencies
├── datasets/
│   ├── raw/               # Raw source data CSV files
│   └── processed/         # Labeled and merged dataset artifacts (goodreads_merged.csv)
└── output/
    ├── shelf_classifier_pipeline.joblib   # Serialized production pipeline
    ├── tfidf_vocab.json                  # Exported TF-IDF vocabulary & class labels
    ├── shelf_classifier.onnx             # Exported ONNX model format
    └── experiments/                      # Hyperparameter tuning CSV reports
```

---

## Setup and Installation

### 1. Environment Initialization

Create and activate a isolated Python 3 virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Dependency Installation

Upgrade `pip` and install all required machine learning and web scraping dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Playwright Browser Installation (Optional for Web Scraping)

If running supplemental web scraping scripts:

```bash
playwright install chromium
```

---

## Pipeline Execution Workflow

Execute the workflow steps sequentially:

### Step 1: Data Preparation and Merging

Clean raw Goodreads data, execute priority-ordered genre tag mapping, and merge primary data with supplemental sources:

```bash
python data_prep.py
```

Outputs the clean, merged dataset artifact to `datasets/processed/goodreads_merged.csv` across 16 target shelf categories.

### Step 2: Hyperparameter Tuning Experiments (Optional)

Run multi-variant model and parameter evaluation without altering production artifacts:

```bash
python tune_experiments.py
```

Evaluates 6 configuration variants (Baseline LinearSVC, C-regularization adjustments, n-gram ranges, vocabulary sizes, MultinomialNB, and LogisticRegression) and writes comparative results to `output/experiments/tuning_summary.csv`.

### Step 3: Train Production Model

Train the production `TfidfVectorizer` + `LogisticRegression` pipeline on the merged dataset:

```bash
python train_model.py
```

Generates diagnostic evaluation metrics (Overall Accuracy, Full Classification Report, 16-Category Confusion Matrix, and 5 Lowest F1 Warning List) and serializes the trained pipeline artifact to `output/shelf_classifier_pipeline.joblib`.

### Step 4: Export Model Metadata and ONNX Representation

Extract feature weights, class label ordering, and export to ONNX for mobile client inference:

```bash
python export_model.py
```

Generates:

1. `output/tfidf_vocab.json`: Class label ordering, TF-IDF feature vocabulary index, and IDF weights for Flutter client vectorization.
2. `output/shelf_classifier.onnx`: Intermediate ONNX format for on-device inference engines.

---

## Mobile App Integration (Flutter)

1. Include `output/tfidf_vocab.json` and `output/shelf_classifier.onnx` (or compiled TFLite models) in the mobile client `assets/` directory.
2. Vectorize input document metadata using the exported vocabulary dictionary.
3. Feed input feature arrays to the on-device inference runtime (`tflite_flutter` or ONNX runtime) to receive shelf category predictions and confidence distributions.
