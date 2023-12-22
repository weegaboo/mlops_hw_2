from typing import Any

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from ..src.models import Model
from ..src.schemas import TrainData, PredictData
from ..src.model_manager import ModelStorageManager
from ..src.controller import Controller


@pytest.fixture
def mock_storage_manager(mocker):
    # Создаем мок для объекта ModelStorageManager
    return mocker.Mock(spec=ModelStorageManager)


@pytest.fixture
def mock_model(mocker, mock_storage_manager):
    # Создаем мок для объекта Model
    model = mocker.Mock(spec=Model)
    model.storage_manager = mock_storage_manager
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
    test_app.post("/train", json={})

    pass

def test_predict_endpoint(test_controller):
    # Тестирование endpoint'а /predict
    # Аналогично, можно установить возвращаемые значения или проверить вызовы
    pass


