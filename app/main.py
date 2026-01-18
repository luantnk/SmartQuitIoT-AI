import os
import uvicorn
import shutil
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool


from app.requests.api_schemas import (
    TextCheckRequest,
    MediaUrlRequest,
    QuitPlanPredictRequest,
    PeakCravingRequest,
    TextToSpeechRequest,
    SummaryRequest,
    DiaryAnalysisRequest,
    ReportChartRequest,
)
from app.services.content_moderation_service import (
    is_text_toxic,
    check_image_url,
    check_video_url,
)
from app.services.audio_service import transcribe_audio_file, text_to_speech_file
from app.services.summary_service import (
    summary_service,
    generate_coach_summary,
    generate_peak_intervention,
)
from app.services.report_service import report_service
from app.services.ai_training_service import (
    load_full_rich_data,
    preprocess_common_features,
    train_success_model,
    train_peak_craving_time_model,
)

import app.models as ai_models

app = FastAPI(
    title="SmartQuitIoT AI Service",
    description="Microservice for AI predictions, content moderation, and audio processing.",
    version="2.2.0",
)
CURRENT_WORKING_DIR = os.getcwd()


def _calculate_daily_risk(req: PeakCravingRequest) -> Dict:
    sess = ai_models.onnx_session_craving
    if sess is None:
        raise HTTPException(status_code=503, detail="Craving Model not loaded")
    current_day = (
        req.day_of_week if req.day_of_week is not None else datetime.now().weekday()
    )
    if current_day > 6:
        current_day = 6

    batch_input = []
    time_labels = []

    for step in range(0, 96):
        hour_float = step / 4.0
        hour_part = int(hour_float)
        minute_part = int((hour_float - hour_part) * 60)
        time_labels.append(f"{hour_part:02d}:{minute_part:02d}")

        batch_input.append(
            [
                float(hour_float),
                float(current_day),
                float(req.ftnd_score),
                float(req.smoke_avg_per_day),
                float(req.age),
                float(req.gender_code),
                float(req.mood_level),
                float(req.anxiety_level),
            ]
        )

    input_name = sess.get_inputs()[0].name
    predictions = sess.run(None, {input_name: np.array(batch_input, dtype=np.float32)})[
        0
    ]
    preds_flat = predictions.flatten().tolist()

    max_val = max(preds_flat)
    peak_index = preds_flat.index(max_val)

    return {
        "predictions": preds_flat,
        "time_labels": time_labels,
        "peak_val": max_val,
        "peak_time": time_labels[peak_index],
        "peak_index": peak_index,
    }


def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "AI Service is ready",
        "models": {
            "success": ai_models.onnx_session_success is not None,
            "craving": ai_models.onnx_session_craving is not None,
            "llm": ai_models.hf_client is not None,
        },
    }


@app.post("/train-models", tags=["AI Training"])
async def trigger_training(background_tasks: BackgroundTasks):
    def train_task():
        print("[INFO] Starting training...")
        df = load_full_rich_data()
        if df is not None and not df.empty:
            processed = preprocess_common_features(df)
            train_success_model(processed)
            train_peak_craving_time_model(processed)
            ai_models.load_onnx_models()
            print("[SUCCESS] Models reloaded.")
        else:
            print("[WARN] No data.")

    background_tasks.add_task(train_task)
    return {"status": "Training started."}


@app.post("/predict-risk/mobile", tags=["Prediction"], summary="For Mobile/FCM")
async def predict_risk_mobile(req: PeakCravingRequest):

    data = _calculate_daily_risk(req)
    intervention_msg = await run_in_threadpool(
        generate_peak_intervention,
        data["peak_time"],
        float(data["peak_val"]),
        req.mood_level,
        req.anxiety_level,
    )
    return {
        "peak_time": data["peak_time"],
        "peak_craving_level": round(data["peak_val"], 2),
        "message": intervention_msg,
    }


@app.post("/predict-risk/dashboard", tags=["Prediction"], summary="For Admin Dashboard")
async def predict_risk_dashboard(req: PeakCravingRequest):
    data = _calculate_daily_risk(req)
    preds = data["predictions"]
    segments = {
        "night_avg": np.mean(preds[0:24]),
        "morning_avg": np.mean(preds[24:48]),
        "afternoon_avg": np.mean(preds[48:72]),
        "evening_avg": np.mean(preds[72:96]),
    }
    worst_segment = max(segments, key=segments.get)
    high_risk_threshold = 7.0
    high_risk_count = sum(1 for x in preds if x >= high_risk_threshold)
    high_risk_duration_minutes = high_risk_count * 15
    avg_daily_risk = np.mean(preds)
    return {
        "overview": {
            "peak_time": data["peak_time"],
            "peak_level": round(data["peak_val"], 2),
            "average_daily_risk": round(avg_daily_risk, 2),
            "risk_status": "CRITICAL" if data["peak_val"] > 8 else "MODERATE",
        },
        "analytics": {
            "worst_time_of_day": worst_segment.replace("_avg", "").title(),
            "high_risk_duration_minutes": high_risk_duration_minutes,
            "segments": {k: round(v, 2) for k, v in segments.items()},
        },
        "chart_data": {
            "labels": data["time_labels"],
            "values": [round(x, 2) for x in preds],
        },
    }


@app.post("/predict-quit-status", tags=["Prediction"])
async def predict_quit_status(req: QuitPlanPredictRequest):
    sess = ai_models.onnx_session_success
    if sess is None:
        raise HTTPException(status_code=503, detail="Success Model not loaded")
    try:
        input_data = np.array([req.features], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: input_data})
        probs = result[1][0] if len(result) > 1 else result[0][0]
        success_prob = (
            probs.get(1, 0.0) if isinstance(probs, (dict, map)) else float(probs)
        )

        return {
            "success_probability": round(success_prob * 100, 2),
            "relapse_risk": round((1.0 - success_prob) * 100, 2),
            "recommendation": (
                "Maintain progress" if success_prob > 0.6 else "Urgent support needed"
            ),
        }
    except Exception as e:
        print(f"[ERROR] Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/check-content", tags=["Content Moderation"])
async def api_check_text(req: TextCheckRequest):
    return {"isToxic": is_text_toxic(req.text), "type": "text"}


@app.post("/check-image-url", tags=["Content Moderation"])
async def api_check_image(req: MediaUrlRequest):
    return {"isToxic": check_image_url(req.url), "type": "image"}


@app.post("/check-video-url", tags=["Content Moderation"])
async def api_check_video(req: MediaUrlRequest):
    return {"isToxic": check_video_url(req.url), "type": "video"}


@app.post("/voice-to-text", tags=["Audio"])
async def api_voice_to_text(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if file.filename else "wav"
    temp_path = os.path.join(CURRENT_WORKING_DIR, f"temp_{uuid.uuid4()}.{ext}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"text": transcribe_audio_file(temp_path), "status": "success"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/text-to-voice", tags=["Audio"])
async def api_text_to_voice(req: TextToSpeechRequest, bg: BackgroundTasks):
    out_path = os.path.join(CURRENT_WORKING_DIR, f"tts_{uuid.uuid4()}.wav")
    try:
        text_to_speech_file(req.text, out_path)
        bg.add_task(cleanup_file, out_path)
        return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        cleanup_file(out_path)
        raise HTTPException(500, str(e))


@app.post("/summarize-week", tags=["Coach"])
async def summarize_week(req: SummaryRequest):
    summary = await run_in_threadpool(
        generate_coach_summary, req.member_name, [l.dict() for l in req.logs]
    )
    return {"status": "success", "summary": summary}


@app.post("/analyze-diary", tags=["Coach"])
async def analyze_diary(req: DiaryAnalysisRequest):
    return await run_in_threadpool(summary_service.analyze_diary_sentiment, req.dict())


@app.post("/generate-report-image", tags=["Visualization"])
async def generate_report_image(req: ReportChartRequest):
    img = await run_in_threadpool(
        report_service.generate_report_image,
        req.logs,
        req.member_name,
        req.start_date,
        req.end_date,
    )
    return {"status": "success", "image_base64": img}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
