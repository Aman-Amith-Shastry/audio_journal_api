import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from pydantic import BaseModel
import librosa


from audio_journal_models import (
    model,          # already loaded in your script
    device,
    preprocess      # your preprocess() function
)

app = FastAPI(
    title="Audio Journal Emotion + Metrics API",
    version="1.0.0"
)


# ---------- Response Model ----------
class InferenceResponse(BaseModel):
    logits: list
    predicted_class: int
    speech_rate: float
    loudness_mean: float
    pitch_mean: float
    energy_variability: float


# ---------- GET Endpoint ----------
@app.get("/predict", response_model=InferenceResponse)
async def predict(audio_url: str = Query(..., description="Public URL of the audio file")):
    """
    Example:
    /predict?audio_url=https://storage.googleapis.com/.../file.m4a
    """

    # Run preprocessing
    sample, metrics = preprocess(audio_url)
    print("Preprocess complete")

    # Model inference
    with torch.no_grad():
        logits = model(
            sample["input_values"].to(device),
            sample["attention_mask"].to(device)
        )

    logits_list = logits.squeeze(0).tolist()
    predicted_class = int(torch.argmax(logits, dim=-1).item())

    # Combine everything into JSON response
    return InferenceResponse(
        logits=logits_list,
        predicted_class=predicted_class,
        speech_rate=metrics["speech_rate"],
        loudness_mean=metrics["loudness_mean"],
        pitch_mean=metrics["pitch_mean"],
        energy_variability=metrics["energy_variability"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )