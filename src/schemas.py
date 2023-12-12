from pydantic import BaseModel
from typing import List, Dict


class TrainData(BaseModel):
    """
    TrainData Model

    This class represents the input data for training a machine learning model.

    Attributes:
        model_type (str): The type of the machine learning model (e.g., 'LogisticRegression', 'RandomForestClassifier').
        hyperparameters (Dict): A dictionary of hyperparameters for the model.
        features (List): A list of feature data used for training.
        labels (List): A list of labels corresponding to the feature data.
    """
    model_type: str
    hyperparameters: Dict
    features: List
    labels: List


class PredictData(BaseModel):
    """
    PredictData Model

    This class represents the input data for making predictions with a trained machine learning model.

    Attributes:
        model_type (str): The type of the machine learning model (e.g., 'LogisticRegression', 'RandomForestClassifier').
        features (List): A list of feature data for making predictions.
    """
    model_type: str
    features: List
