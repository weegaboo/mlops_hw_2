import pytest

from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.models import Model
from src.schemas import TrainData, PredictData
from src.controller import Controller


@pytest.fixture
def mock_model(mocker):
    # Создаем мок для объекта Model
    model = mocker.Mock(spec=Model)
    # model.storage_manager = mock_storage_manager
    model.train.return_value = {'message': 'Model trained successfully', 'model_type': 'test'}
    model.predict.return_value = {'prediction': 'test'}
    return model

@pytest.fixture
def test_controller(mock_model):
    # Создаем контроллер с моковым Model
    return Controller(mock_model)

@pytest.fixture
def test_app(test_controller):
    app = FastAPI()
    router = test_controller.configure_routes()
    app.include_router(router)
    return TestClient(app)

def test_train_endpoint(test_app):
    # Тестирование endpoint'а /train
    # Здесь можно установить возвращаемые значения или проверить вызовы методов
    data = {
        "model_type": "test",
        "hyperparameters": {"test": 100},
        "features": [[1, 2, 3], [4, 5, 6]],
        "labels": [0, 1]
    }
    response = test_app.post("/train", json=data)
    assert response.status_code == 200
    assert response.json() == {
        'message': 'Model trained successfully',
        'model_type': 'test'
    }

def test_predict_endpoint(test_app):
    # Тестирование endpoint'а /predict
    # Аналогично, можно установить возвращаемые значения или проверить вызовы
    data = {"model_type": "TestType", "features": [1, 2, 3]}
    response = test_app.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {'prediction': 'test'}
