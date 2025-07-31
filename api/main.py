from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import subprocess

app = FastAPI()

# Enable CORS (for frontend interaction later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/grade")
async def grade_card(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded image
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the model using your terminal script
    try:
        result = subprocess.check_output(
            ["python", "grade_my_card.py", file_path],
            stderr=subprocess.STDOUT,
            cwd=".."
        ).decode("utf-8")
        
        # Extract prediction
        prediction_line = next((line for line in result.splitlines() if "Predicted Grade:" in line), "Prediction not found")
        return JSONResponse({"result": prediction_line.strip()})
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": e.output.decode("utf-8")}, status_code=500)
