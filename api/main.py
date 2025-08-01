from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import io

app = FastAPI()

# Enable CORS (for frontend interaction later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model once on startup
MODEL_PATH = "psa_model.h5"  # Update this if your model path differs
model = load_model(MODEL_PATH)

# Prediction logic
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Resize to match training input
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

@app.post("/grade")
async def grade_card(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_bytes = await file.read()
        processed = preprocess_image(image_bytes)

        # Predict
        prediction = model.predict(processed)
        predicted_grade = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse({
            "predicted_grade": predicted_grade,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

