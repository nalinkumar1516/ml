
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student performance using Machine Learning",
    version="1.0"
)

# Load trained ML model
try:
    data = joblib.load("models/v2.pkl")
    model = data["model"]
    features = data["features"]

    logger.info("ML model loaded successfully")

except Exception as e:
    logger.exception("Error loading ML model")
    raise RuntimeError("Unable to load the trained model") from e


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
    try:
        # Prepare input data
        input_values = {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_marks": previous_marks
        }

        input_data = pd.DataFrame([input_values])

        # Arrange columns according to training features
        input_data = input_data.reindex(columns=features)

        # Make prediction
        prediction = model.predict(input_data)

        if len(prediction) == 0:
            raise ValueError("Model returned an empty prediction")

        result = str(prediction[0])

        logger.info("Prediction completed successfully: %s", result)

        return {
            "prediction_result": result
        }

    except ValueError as e:
        logger.exception("Invalid prediction input")
        raise HTTPException(
            status_code=400,
            detail="Invalid prediction input"
        ) from e

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please check the server logs."
        ) from e
