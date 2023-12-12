from fastapi import APIRouter
from .models import Model
from .schemas import TrainData, PredictData


class Controller:
    """
    Controller Class

    Controller class provides FastAPI API routes for model training, predictions,
    and model management.

    Args:
        processor (Model): An instance of the Model class used for data processing and predictions.

    Methods:
        configure_routes(): Configures and returns an APIRouter with routes for training and predictions.

    Attributes:
        processor (Model): The Model instance used for data processing.
    """
    def __init__(self, processor: Model):
        self.processor = processor

    def configure_routes(self):
        """
        configure_routes Method

        Creates and configures an APIRouter with FastAPI routes for training and predictions.

        Returns:
            APIRouter: Configured APIRouter with training and prediction routes.
        """
        api_router = APIRouter()

        @api_router.post('/train')
        async def train(data: TrainData):
            """
            Train Endpoint

            Trains the model based on the provided training data.

            Args:
                data (TrainData): Input data for model training.

            Returns:
                dict: The training result.
            """
            return self.processor.train(data)

        @api_router.post('/predict')
        async def predict(data: PredictData):
            """
            Predict Endpoint

            Performs predictions based on the provided input data.

            Args:
                data (PredictData): Input data for making predictions.

            Returns:
                dict: The prediction result.
            """
            return self.processor.predict(data)

        @api_router.get('/models')
        async def get_models():
            """
            Get Models Endpoint

            Retrieves a list of available models for predictions.

            Returns:
                list: List of available models.
            """
            return self.processor.storage_manager.get_available_for_predict()

        @api_router.delete('/delete/{name}')
        async def remove_model(name: str):
            """
            Delete Model Endpoint

            Deletes a model by its name.

            Args:
                name (str): The name of the model to delete.

            Returns:
                dict: The result of the model deletion operation.
            """
            return self.processor.storage_manager.delete(name)

        return api_router


handler = Model()
controller = Controller(handler)
router = controller.configure_routes()
