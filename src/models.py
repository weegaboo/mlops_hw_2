from typing import Dict, Any
from fastapi import HTTPException
from .schemas import TrainData, PredictData
from .model_manager import ModelStorageManager, HyperparametersManager


class Model:
    """
    Model Class

    The Model class represents a machine learning model and provides methods for training and prediction.

    Attributes:
        storage_manager (ModelStorageManager): An instance of ModelStorageManager for model storage.
    """
    def __init__(self):
        self.storage_manager = ModelStorageManager

    def train(self, data: TrainData) -> Dict[str, Any]:
        """
        Train Method

        Trains a machine learning model based on the provided training data and hyperparameters.

        Args:
            data (TrainData): Input data for model training.

        Returns:
            Dict[str, Any]: A dictionary with a success message and the model type.

        Raises:
            HTTPException: If an error occurs during model training.
        """
        model_class = self.storage_manager.fetch_training_model(data.model_type)
        manager = HyperparametersManager(model_class, data.hyperparameters)
        try:
            manager.model.fit(data.features, data.labels)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        self.storage_manager.save(manager.model, data.model_type)
        return {
            'message': 'Model trained successfully',
            'model_type': data.model_type
        }

    def predict(self, data: PredictData) -> Dict[str, int]:
        """
        Predict Method

        Performs predictions using a trained machine learning model.

        Args:
            data (PredictData): Input data for making predictions.

        Returns:
            Dict[str, int]: A dictionary with the prediction result.

        Raises:
            HTTPException: If an error occurs during prediction.
        """
        model = self.storage_manager.fetch_predicting_model(data.model_type)
        try:
            prediction = model.predict(data.features)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return {'prediction': prediction.tolist()}
