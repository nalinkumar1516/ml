
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model dictionary
data = joblib.load("models/v2.pkl")

# Extract actual trained ML model
model = data["model"]

# Get training feature names
features = data["features"]


@app.get("/")
def home():
    return {
        "message": "Welcome to the Student Performance Prediction API!"
    }


@app.post("/predict")
def predict(
    study_hours: float,
    attendance: float,
    previous_marks: float
):
    input_values = {
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks
    }

    # Arrange input according to model training features
    input_data = pd.DataFrame([input_values])

    input_data = input_data.reindex(columns=features)

    prediction = model.predict(input_data)

    return {
        "prediction_result": float(prediction[0])
    }
