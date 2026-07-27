# The Shelf — Training Pipeline (`the-shelf-training`)

Machine Learning training pipeline for **The Shelf** — a Flutter mobile app that organizes books, PDFs, and documents into auto-sorted shelves using an on-device text classifier.

---

## 📁 Repository Structure

```text
the-shelf-training/
├── .gitignore
├── requirements.txt
├── README.md
├── scrape_data.py        # Web scraper (Playwright + stealth) for non-API web sources (Goodreads)
├── fetch_api_data.py     # Data fetcher for free public APIs (Open Library, Jikan / MyAnimeList)
├── train_model.py        # Model training pipeline (TF-IDF + Linear SVM / Naive Bayes)
├── export_model.py       # Model exporter converting trained model to ONNX & TFLite for Flutter
├── datasets/             # Directory for storing raw & processed dataset JSON/CSV files
│   └── .gitkeep
└── output/               # Directory for trained models, vocabularies, and exported .tflite files
    └── .gitkeep
```

---

## 🛠️ Setup & Installation

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Playwright Browsers (for web scraping)

```bash
playwright install chromium
```

---

## 🚀 Pipeline Workflow

Run the pipeline steps in the following order:

### Step 1: Data Collection

Fetch structured book and document data from public APIs or web sources:

```bash
# Option A: Call free public APIs (Open Library & Jikan / MyAnimeList) [Recommended]
python fetch_api_data.py

# Option B: Scrape web sources (e.g., Goodreads) with Playwright stealth
python scrape_data.py
```

Collected dataset files are saved to `datasets/`.

### Step 2: Train Classifier

Train the text classification model (TF-IDF vectorizer + Linear SVM) on collected data:

```bash
python train_model.py
```

Outputs the trained model pipeline (`shelf_classifier_pipeline.joblib`) into `output/`.

### Step 3: Export to Mobile (TFLite)

Convert and package the model for mobile on-device inference:

```bash
python export_model.py
```

Generates ONNX intermediate models (`shelf_classifier.onnx`), vocabulary JSON metadata (`tfidf_vocab.json`), and mobile TFLite models (`shelf_classifier.tflite`) in `output/`.

---

## 📱 Mobile Integration (Flutter)

1. Copy `output/shelf_classifier.tflite` and `output/tfidf_vocab.json` into your Flutter app's `assets/` directory.
2. Use [`tflite_flutter`](https://pub.dev/packages/tflite_flutter) to load the TFLite model on iOS and Android.
3. Preprocess PDF text / document metadata using the exported `tfidf_vocab.json` dictionary before feeding input tensors to the model.
