"""
Load the trained machine learning model.
"""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "fare_model.joblib"


def load_model():
    """
    Load the trained model from disk.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)