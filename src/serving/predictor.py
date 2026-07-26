"""
Shared prediction utilities.
"""

import pandas as pd

from src.serving.model_loader import load_model


def predict(dataframe: pd.DataFrame):
    """
    Generate predictions using the trained model.
    """

    model = load_model()

    predictions = model.predict(dataframe)

    return predictions