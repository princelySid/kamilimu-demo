import os
import pickle

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from src.operations.example import convert_data
from src.schemas import PredictData

load_dotenv()
app = FastAPI()
app = FastAPI(
    title="Kamilimu Demo", description="API for board game ml model", version="1.0"
)


@app.post("/predict", tags=["predictions"])
async def get_prediction(data: PredictData):
    ml_model = pickle.load(os.getenv("ML_MODEL"))
    data = convert_data(data.dict())
    prediction = ml_model.predict(data)
    return {"prediction": prediction}
