import os
import io
import pickle

from typing import Dict, List, Any, Type
from minio import Minio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fastapi import HTTPException


class HyperparametersManager:
    """
    HyperparametersManager Class

    The HyperparametersManager class is responsible for managing hyperparameters and creating model instances
    with the provided hyperparameters.

    Args:
        estimator (type): The type of machine learning estimator (e.g., LogisticRegression, RandomForestClassifier).
        hyperparameters (Dict[str, Any]): A dictionary of hyperparameters to be used for model initialization.

    Methods:
        check_hyperparameters(estimator, input_): Static method to check if provided hyperparameters match the estimator.
    """
    def __init__(self, estimator: Type, hyperparameters: Dict):
        self.check_hyperparameters(estimator, hyperparameters)
        self.model = estimator(**hyperparameters)

    @staticmethod
    def check_hyperparameters(estimator: Type, input_: Dict[str, Any]):
        """
        check_hyperparameters Static Method

        Checks if provided hyperparameters match the estimator's parameter names.

        Args:
            estimator (type): The type of machine learning estimator.
            input_ (Dict[str, Any]): A dictionary of hyperparameters to be checked.

        Raises:
            HTTPException: If provided hyperparameters do not match the model type.
        """
        params = estimator._get_param_names()
        params_difference = set(input_).difference(params)
        if len(params_difference):
            raise HTTPException(
                status_code=400,
                detail="Provided hyperparameters do not match the model type."
            )


class ModelStorageManager:
    """
    ModelStorageManager Class

    The ModelStorageManager class provides functionality for managing machine learning models,
    including saving, loading, and deleting models.

    Attributes:
        save_dir (str): The directory where models are saved.
        models (Dict[str, type]): A dictionary mapping model names to their corresponding types.

    Methods:
        fetch_training_model(name: str) -> Any: Retrieves the training model type by name.
        fetch_predicting_model(name: str) -> Any: Retrieves a trained model by name for predictions.
        get_available_for_predict() -> Dict: Returns available models for predictions.
        save(model, name: str): Saves a trained model to the specified name.
        open(name: str) -> Any: Opens a saved model by name.
        delete(name: str): Deletes a saved model by name.
    """
    models = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'SVC': SVC
    }
    buckets = {"trained-models", "features", "labels"}

    def __init__(self, endpoint: str, access_key: str, secret_key: str):
        self.client = Minio(endpoint, access_key, secret_key, secure=False)
        server_buckets = [bucket.name for bucket in self.client.list_buckets()]
        for new_bucket in self.buckets.difference(server_buckets):
            self.client.make_bucket(bucket_name=new_bucket)


    @classmethod
    def __fetch_check(cls, fetch_from, name: str):
        if name not in fetch_from:
            raise HTTPException(status_code=404, detail="Model not found")

    def fetch_trained_model_names(self) -> List[str]:
        trained_models = self.client.list_objects(bucket_name="trained-models")
        return [model.object_name for model in trained_models]

    @classmethod
    def fetch_training_model(cls, name: str) -> Any:
        """
        fetch_training_model Method

        Retrieves the training model type by name.

        Args:
            name (str): The name of the model.

        Returns:
            Any: The training model type.

        Raises:
            HTTPException: If the model is not found.
        """
        cls.__fetch_check(cls.models, name)
        return cls.models[name]

    def fetch_predicting_model(self, name: str) -> Any:
        """
        fetch_predicting_model Method

        Retrieves a trained model by name for predictions.

        Args:
            name (str): The name of the model.

        Returns:
            Any: The trained model for predictions.

        Raises:
            HTTPException: If the model is not found.
        """
        trained_models = self.fetch_trained_model_names()
        self.__fetch_check(trained_models, name)
        return self.open(bucket_name="trained-models", object_name=name)

    def get_available_for_predict(self) -> Dict:
        """
        get_available_for_predict Method

        Returns available models for predictions.

        Returns:
            Dict: A dictionary containing available models for predictions.
        """
        return {'models': self.fetch_trained_model_names()}

    def save(self, item, bucket_name: str, object_name: str):
        """
        save Method

        Saves trained model or data to the specified name.

        Args:
            item: data or model
            bucket_name (str):
            object_name (str):
        """

        bytes = pickle.dumps(item)
        self.client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(bytes),
            length=len(bytes)
        )
        #ДОБАВИТЬ DVC

    def open(self, bucket_name: str, object_name: str) -> Any:
        """
        open Method

        Opens a saved model by name.

        Args:
            bucket_name (str): bucket name
            object_name (str): object name

        Returns:
            Any: The opened machine learning model.
        """

        answer = self.client.get_object(bucket_name=bucket_name, object_name=object_name)
        if answer.status != 200:
            raise HTTPException(status_code=answer.status, detail="Server open error")
        return pickle.loads(answer.data)

    def delete(self, bucket_name: str, object_name: str):
        """
        delete Method

        Deletes a saved model by name.

        Args:
            bucket_name (str): bucket name
            object_name (str): object name

        Returns:
            Dict: A dictionary with a message indicating the success of the model deletion.
        """
        self.client.remove_object(bucket_name=bucket_name, object_name=object_name)
        return {'message': 'Model removed successfully'}
