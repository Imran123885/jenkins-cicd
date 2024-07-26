from sklearn.pipeline import Pipeline
from prediction_model.config import config
import prediction_model.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

classification_pipeline = Pipeline(
    [
        ('Mean Imputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('Mode Imputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('Domain Processing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY, variable_to_add=config.FEATURE_TO_ADD)),
        ('Drop Features', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('Label Encoder', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('Log Transformation', pp.LogProcessing(variables=config.LOG_FEATURES)),
        ('Min Max Scaler', MinMaxScaler()),
        ('Logistic Classifier', LogisticRegression(random_state=0))
    ]
)
