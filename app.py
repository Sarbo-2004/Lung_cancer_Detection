# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, predict_image

app = FastAPI()
model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, confidence, probs = predict_image(image_bytes, model)

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "probabilities": dict(zip(["lung_aca", "lung_n", "lung_scc"], probs))
    }
