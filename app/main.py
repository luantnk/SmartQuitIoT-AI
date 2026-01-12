import os
import uvicorn
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.content_moderation import is_text_toxic, check_image_url, check_video_url

app = FastAPI(title="SmartQuitIoT AI Service")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_WORKING_DIR = os.getcwd()

candidate_paths = [
    os.path.join(CURRENT_WORKING_DIR, "app", "models", "smartquit_model.onnx"),
    os.path.join(BASE_DIR, "app", "models", "smartquit_model.onnx"),
    os.path.join(BASE_DIR, "models", "smartquit_model.onnx"),
    "smartquit_model.onnx"
]

sess = None
MODEL_PATH = None

for path in candidate_paths:
    if os.path.exists(path):
        try:
            sess = rt.InferenceSession(path)
            MODEL_PATH = path
            print(f"AI Model loaded successfully from: {path}")
            break
        except Exception as e:
            print(f"Found file but failed to load at {path}: {e}")

if sess is None:
    print("CRITICAL: Could not find smartquit_model.onnx in any expected location.")
    print(f"Searched in: {candidate_paths}")



class TextCheckRequest(BaseModel):
    text: str


class MediaUrlRequest(BaseModel):
    url: str


class QuitPlanPredictRequest(BaseModel):
    features: list[float]


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "AI Service is ready",
        "model_loaded": sess is not None,
        "model_path": MODEL_PATH if MODEL_PATH else "Not Found"
    }


@app.post("/predict-quit-status", tags=["Plan Prediction"])
def predict_quit_status(req: QuitPlanPredictRequest):
    if sess is None:
        raise HTTPException(status_code=503, detail="Prediction model not found on server")
    try:
        input_data = np.array([req.features], dtype=np.float32)

        input_name = sess.get_inputs()[0].name

        result = sess.run(None, {input_name: input_data})
        success_prob = float(result[1][0][1])
        relapse_risk = 1.0 - success_prob

        return {
            "success_probability": round(success_prob * 100, 2),
            "relapse_risk": round(relapse_risk * 100, 2),
            "recommendation": "Maintain progress" if success_prob > 0.6 else "Urgent support needed"
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")




@app.post("/check-content", tags=["Content Moderation"])
def api_check_text(req: TextCheckRequest):
    try:
        toxic = is_text_toxic(req.text)
        return {"isToxic": toxic, "type": "text"}
    except Exception as e:
        print(f"Error checking text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-image-url", tags=["Content Moderation"])
def api_check_image(req: MediaUrlRequest):
    try:
        nsfw = check_image_url(req.url)
        return {"isToxic": nsfw, "type": "image"}
    except Exception as e:
        print(f"Error checking image: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/check-video-url", tags=["Content Moderation"])
def api_check_video(req: MediaUrlRequest):
    try:
        nsfw = check_video_url(req.url)
        return {"isToxic": nsfw, "type": "video"}
    except Exception as e:
        print(f"Error checking video: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)