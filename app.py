from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("models/model.pkl", "rb"))

@app.get("/predict")
def predict(area: float, bedrooms: int):
    data = np.array([[area, bedrooms]])
    price = model.predict(data)[0]
    return {"price": float(price)}