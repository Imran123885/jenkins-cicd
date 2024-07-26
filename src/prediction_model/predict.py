import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os 
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline

classification_pipeline = load_pipeline(config.MODEL_NAME)

def generate_pred(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1, 'Y', 'N')
    result = {'Predictions': output}
    return result