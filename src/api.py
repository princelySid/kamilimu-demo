import os
import pickle

from dotenv import load_dotenv
from fastapi import FastAPI

from src.operations.example import convert_data
from src.schemas import PredictData

load_dotenv()
app = FastAPI()
app = FastAPI(
    title="Kamilimu Demo", description="API for board game ml model", version="1.0"
)


@app.get("/something")
def something():
    return "I am going to return a string"


@app.post("/predict", tags=["predictions"])
def get_prediction(data: PredictData):
    with open(os.getenv("ML_MODEL"), "rb") as f:
        ml_model = pickle.load(f)
    data = convert_data(data.dict())
    prediction = ml_model.predict(data)
    return {"prediction": prediction[0]}
