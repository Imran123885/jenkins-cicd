import pandas as pd
import numpy as np

from pathlib import Path
import os 
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
import prediction_model.processing.preprocessing as pp
import prediction_model.pipeline as pipe

def train():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({'N':0, 'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES],train_y)
    save_pipeline(pipe.classification_pipeline)
if __name__=="__main__":
    train()