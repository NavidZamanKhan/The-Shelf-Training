"""
export_model.py
---------------
Model export & conversion script for 'The Shelf' mobile app (Flutter).

Scikit-learn models (TF-IDF + Linear SVM / Naive Bayes) cannot be directly converted to TFLite 
in a single step. This script documents and scaffolds the standard multi-step conversion strategies:

Strategy 1: Scikit-learn -> ONNX (via skl2onnx) -> TFLite (via onnx2tf / TensorFlow Lite Converter)
Strategy 2: Keras / TensorFlow Re-implementation (TextVectorization + Dense layer) -> TFLite
Strategy 3: Flutter-side TF-IDF Preprocessing + TFLite Classifier (Dense / Softmax model)

NOTE:
    - For Flutter on-device inference via `tflite_flutter` or `flutter_tflite`, Strategy 3 
      or Strategy 2 is often the most reliable because string tokenization inside TFLite runtime 
      requires custom ops unless TensorFlow Text / Keras TextVectorization is used.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_MODEL_PATH = OUTPUT_DIR / "shelf_classifier_pipeline.joblib"
ONNX_OUTPUT_PATH = OUTPUT_DIR / "shelf_classifier.onnx"
TFLITE_OUTPUT_PATH = OUTPUT_DIR / "shelf_classifier.tflite"
VOCAB_OUTPUT_PATH = OUTPUT_DIR / "tfidf_vocab.json"


def export_vocabulary_and_labels(pipeline: Any) -> None:
    """
    Extracts TF-IDF vocabulary, idf weights, and class labels from the trained scikit-learn pipeline 
    so Flutter can perform lightweight on-device vectorization if needed.
    
    Args:
        pipeline: Trained Scikit-learn Pipeline object.
    """
    logger.info("Extracting vocabulary and labels for Flutter on-device vectorizer...")
    
    try:
        tfidf = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]

        vocab = tfidf.vocabulary_  # word -> feature_index mapping
        idf_weights = tfidf.idf_.tolist()
        classes = classifier.classes_.tolist()

        metadata = {
            "classes": classes,
            "vocabulary": vocab,
            "idf": idf_weights,
            "max_features": tfidf.max_features,
            "ngram_range": list(tfidf.ngram_range),
            "stop_words": "english"
        }

        with open(VOCAB_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved vocabulary ({len(vocab)} terms) and class labels to {VOCAB_OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"Error extracting metadata from pipeline: {e}")


def convert_sklearn_to_onnx(pipeline: Any, save_path: Path) -> Optional[Path]:
    """
    Converts trained scikit-learn pipeline into ONNX intermediate format using `skl2onnx`.
    
    Args:
        pipeline: Trained Scikit-learn Pipeline object.
        save_path (Path): Target path for `.onnx` model file.
        
    Returns:
        Optional[Path]: Path to saved ONNX file if successful.
    """
    logger.info("Converting scikit-learn pipeline to ONNX format via skl2onnx...")
    
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType
        
        # Define input shape: 1D batch of strings
        initial_type = [("input_text", StringTensorType([None, 1]))]
        
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=15
        )
        
        with open(save_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        logger.info(f"Successfully converted and saved ONNX model to {save_path}")
        return save_path

    except ImportError:
        logger.warning(
            "skl2onnx package not installed or missing dependencies. "
            "Install via `pip install skl2onnx onnxruntime` to enable ONNX conversion."
        )
    except Exception as e:
        logger.error(f"Failed ONNX conversion: {e}")
        
    return None


def convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> Optional[Path]:
    """
    Converts ONNX model to TFLite format using `onnx2tf` or TensorFlow Lite Converter.
    
    Args:
        onnx_path (Path): Path to source `.onnx` file.
        tflite_path (Path): Destination path for `.tflite` file.
        
    Returns:
        Optional[Path]: Path to saved TFLite file if successful.
    """
    logger.info(f"Converting ONNX model ({onnx_path}) to TFLite format ({tflite_path})...")
    
    # TODO: Implement conversion step via onnx2tf or tf2onnx / TFLiteConverter
    # Example shell command or python call:
    # import onnx2tf
    # onnx2tf.convert(input_onnx_file_path=str(onnx_path), output_folder_path=str(OUTPUT_DIR))

    logger.info(
        "Placeholder: ONNX -> TFLite conversion step.\n"
        "  Recommended tool: `onnx2tf -i output/shelf_classifier.onnx -o output/`\n"
        "  Alternatively, train Keras model with `TextVectorization` and export via `tf.lite.TFLiteConverter`."
    )
    
    return None


def main() -> None:
    """
    Main export execution function.
    """
    logger.info("Starting model export workflow for 'The Shelf' mobile app...")
    
    if not INPUT_MODEL_PATH.exists():
        logger.error(f"Input model file not found at {INPUT_MODEL_PATH}. Please run `train_model.py` first.")
        return

    # 1. Load scikit-learn trained model pipeline
    logger.info(f"Loading trained model from {INPUT_MODEL_PATH}")
    pipeline = joblib.load(INPUT_MODEL_PATH)

    # 2. Extract vocabulary & class labels for Flutter client metadata
    export_vocabulary_and_labels(pipeline)

    # 3. Step 1: Convert to ONNX intermediate representation
    onnx_file = convert_sklearn_to_onnx(pipeline, ONNX_OUTPUT_PATH)

    # 4. Step 2: Convert ONNX representation to TFLite
    if onnx_file and onnx_file.exists():
        convert_onnx_to_tflite(onnx_file, TFLITE_OUTPUT_PATH)
    else:
        logger.info("Skipping TFLite conversion step as ONNX model generation was omitted.")

    logger.info("Export workflow complete.")


if __name__ == "__main__":
    main()
