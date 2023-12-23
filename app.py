import uvicorn

from fastapi import FastAPI
from src.models import Model
from src.schemas import TrainData, PredictData
from src.model_manager import ModelStorageManager
from src.controller import Controller


storage_manager = ModelStorageManager(
    endpoint=os.environ.get("ENDPOINT"),
    access_key=os.environ.get("ACCESS_KEY"),
    secret_key=os.environ.get("SECRET_KEY")
)
processor = Model(storage_manager)
controller = Controller(processor)
router = controller.configure_routes()

app = FastAPI()
app.include_router(router)
