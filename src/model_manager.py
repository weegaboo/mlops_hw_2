import os
import pickle

from typing import Dict, Any, Type
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from fastapi import HTTPException
from .config_reader import config


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
    save_dir = config.save_dir
    models = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'SVC': SVC
    }

    @classmethod
    def __fetch_check(cls, fetch_from, name: str):
        if name not in fetch_from:
            raise HTTPException(status_code=404, detail="Model not found")

    @classmethod
    def __join_save_dir_to_name(cls, name: str):
        return os.path.join(cls.save_dir, f'{name}.pkl')

    @classmethod
    def __get_names_from_save_dir(cls):
        return [file.split(sep='.')[0] for file in os.listdir(cls.save_dir)]

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

    @classmethod
    def fetch_predicting_model(cls, name: str) -> Any:
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
        trained_models = cls.__get_names_from_save_dir()
        cls.__fetch_check(trained_models, name)
        return cls.open(name)

    @classmethod
    def get_available_for_predict(cls) -> Dict:
        """
        get_available_for_predict Method

        Returns available models for predictions.

        Returns:
            Dict: A dictionary containing available models for predictions.
        """
        trained_models = [
            file
            for file in cls.__get_names_from_save_dir()
            if file in cls.models
        ]
        return {'models': trained_models}

    @classmethod
    def save(cls, model, name: str):
        """
        save Method

        Saves a trained model to the specified name.

        Args:
            model: The trained machine learning model.
            name (str): The name under which the model should be saved.
        """
        path = cls.__join_save_dir_to_name(name)
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    @classmethod
    def open(cls, name: str) -> Any:
        """
        open Method

        Opens a saved model by name.

        Args:
            name (str): The name of the model to open.

        Returns:
            Any: The opened machine learning model.
        """
        path = cls.__join_save_dir_to_name(name)
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model

    @classmethod
    def delete(cls, name: str):
        """
        delete Method

        Deletes a saved model by name.

        Args:
            name (str): The name of the model to delete.

        Returns:
            Dict: A dictionary with a message indicating the success of the model deletion.
        """
        trained_models = cls.__get_names_from_save_dir()
        cls.__fetch_check(trained_models, name)
        path = cls.__join_save_dir_to_name(name)
        os.remove(path)
        return {'message': 'Model removed successfully'}
