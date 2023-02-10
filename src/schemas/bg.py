from pydantic import BaseModel


class PredictData(BaseModel):
    average: float
    bayes_average: float
    users_rates: float
