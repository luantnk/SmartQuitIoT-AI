import os
import uvicorn
import shutil
import uuid
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

# Imports
from app.requests.api_schemas import (
    TextCheckRequest, MediaUrlRequest, QuitPlanPredictRequest,
    PeakCravingRequest, TextToSpeechRequest, SummaryRequest,
    DiaryAnalysisRequest, ReportChartRequest
)
from app.services.content_moderation_service import is_text_toxic, check_image_url, check_video_url
from app.services.audio_service import transcribe_audio_file, text_to_speech_file
from app.services.summary_service import summary_service, generate_coach_summary
from app.services.report_service import report_service
from app.services.ai_training_service import (
    load_full_rich_data, preprocess_common_features,
    train_success_model, train_peak_craving_time_model
)

# Alias to avoid variable collision
import app.models as ai_models

# --- CONFIGURATION ---
app = FastAPI(
    title="SmartQuitIoT AI Service",
    description="Microservice for AI predictions, content moderation, and audio processing.",
    version="2.0.0"
)
CURRENT_WORKING_DIR = os.getcwd()


# --- UTILS ---
def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)


#  SYSTEM & HEALTH
@app.get("/health", tags=["System"], summary="Check API Health Status")
async def health_check():
    """
    Returns the status of the AI service and checks if models are loaded in memory.
    """
    return {
        "status": "AI Service is ready",
        "models_dir": ai_models.MODELS_DIR,
        "success_model": ai_models.onnx_session_success is not None,
        "craving_model": ai_models.onnx_session_craving is not None,
    }



#  AI TRAINING
def background_training_task():
    print("[INFO] Starting background training...")
    try:
        df = load_full_rich_data()
        if df is not None and not df.empty:
            processed_df = preprocess_common_features(df)
            train_success_model(processed_df)
            train_peak_craving_time_model(processed_df)

            print("[INFO] Training done. Reloading models into memory...")
            ai_models.load_onnx_models()
            print("[SUCCESS] Models reloaded and ready.")
        else:
            print("[WARNING] No data for training.")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")


@app.post("/train-models", tags=["AI Training"], summary="Trigger Model Retraining")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers a background task to fetch data from MariaDB and retrain the XGBoost models.
    Automatically reloads the models into memory upon completion.
    """
    background_tasks.add_task(background_training_task)
    return {"status": "Training started in background. Models will reload automatically."}



# Prediction
@app.post("/predict-quit-status", tags=["Prediction"], summary="Predict Success Probability")
async def predict_quit_status(req: QuitPlanPredictRequest):
    """
    Predicts the probability of a user succeeding in their quit plan based on static features.
    """
    sess = ai_models.onnx_session_success
    if sess is None:
        raise HTTPException(status_code=503, detail="Success Model not loaded")

    try:
        input_data = np.array([req.features], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: input_data})

        probs = result[1][0] if len(result) > 1 else result[0][0]
        success_prob = probs.get(1, 0.0) if isinstance(probs, (dict, map)) else float(probs)

        return {
            "success_probability": round(success_prob * 100, 2),
            "relapse_risk": round((1.0 - success_prob) * 100, 2),
            "recommendation": "Maintain progress" if success_prob > 0.6 else "Urgent support needed"
        }
    except Exception as e:
        print(f"[ERROR] Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/predict-peak-craving", tags=["Prediction"], summary="Predict Peak Craving Time")
async def predict_peak_craving(req: PeakCravingRequest):
    """
    Predicts the exact time (15-minute intervals) when a user is most likely to crave smoking.
    Returns the peak time (e.g., '14:45') and the calculated risk level.
    """
    sess = ai_models.onnx_session_craving
    if sess is None:
        raise HTTPException(status_code=503, detail="Craving Model not loaded")

    try:
        # Normalize Day of Week
        current_day = req.day_of_week
        if current_day is None:
            current_day = datetime.now().weekday()
        elif current_day > 6:
            current_day = 6


        batch_input = []
        time_labels = []


        for step in range(0, 24 * 4):  # 96 intervals
            hour_float = step / 4.0

            # Format label for response (e.g., 22.5 -> "22:30")
            hour_part = int(hour_float)
            minute_part = int((hour_float - hour_part) * 60)
            time_str = f"{hour_part:02d}:{minute_part:02d}"
            time_labels.append(time_str)

            batch_input.append([
                float(hour_float),  # Using float for minute precision
                float(current_day),
                float(req.ftnd_score),
                float(req.smoke_avg_per_day),
                float(req.age),
                float(req.gender_code),
                float(req.mood_level),
                float(req.anxiety_level)
            ])

        # Run Inference on all 96 time slots
        input_name = sess.get_inputs()[0].name
        predictions = sess.run(None, {input_name: np.array(batch_input, dtype=np.float32)})[0]

        # Flatten and Find Max
        preds_flat = predictions.flatten().tolist()
        max_val = max(preds_flat)
        peak_index = preds_flat.index(max_val)

        peak_time_str = time_labels[peak_index]  # Get the specific time label

        return {
            "peak_time": peak_time_str,  # e.g., "22:15"
            "peak_craving_level": round(max_val, 2),
            "message": f"High risk detected at {peak_time_str}. Be prepared!",
            # Optional: Return a simplified chart data (e.g., just hourly averages if 96 points is too much)
            "data_points": 96,
            "chart_data": preds_flat
        }

    except Exception as e:
        print(f"[ERROR] Peak Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


#  CONTENT MODERATION
@app.post("/check-content", tags=["Content Moderation"], summary="Detect Toxic Text")
async def api_check_text(req: TextCheckRequest):
    return {"isToxic": is_text_toxic(req.text), "type": "text"}


@app.post("/check-image-url", tags=["Content Moderation"], summary="Detect NSFW Image")
async def api_check_image(req: MediaUrlRequest):
    return {"isToxic": check_image_url(req.url), "type": "image"}


@app.post("/check-video-url", tags=["Content Moderation"], summary="Detect NSFW Video")
async def api_check_video(req: MediaUrlRequest):
    return {"isToxic": check_video_url(req.url), "type": "video"}


#  AUDIO PROCESSING
@app.post("/voice-to-text", tags=["Audio"], summary="Transcribe Audio (Whisper)")
async def api_voice_to_text(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if file.filename else "wav"
    temp_path = os.path.join(CURRENT_WORKING_DIR, f"temp_{uuid.uuid4()}.{ext}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"text": transcribe_audio_file(temp_path), "status": "success"}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)


@app.post("/text-to-voice", tags=["Audio"], summary="Generate Speech (SpeechT5)")
async def api_text_to_voice(req: TextToSpeechRequest, bg: BackgroundTasks):
    out_path = os.path.join(CURRENT_WORKING_DIR, f"tts_{uuid.uuid4()}.wav")
    try:
        text_to_speech_file(req.text, out_path)
        bg.add_task(cleanup_file, out_path)
        return FileResponse(out_path, media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        cleanup_file(out_path)
        raise HTTPException(500, str(e))



#  COACH & REPORT
@app.post("/summarize-week", tags=["Coach"], summary="Generate Weekly Summary")
async def summarize_week(req: SummaryRequest):
    summary = await run_in_threadpool(generate_coach_summary, req.member_name, [l.dict() for l in req.logs])
    return {"status": "success", "summary": summary}


@app.post("/analyze-diary", tags=["Coach"], summary="Analyze Daily Sentiment")
async def analyze_diary(req: DiaryAnalysisRequest):
    return await run_in_threadpool(summary_service.analyze_diary_sentiment, req.dict())


@app.post("/generate-report-image", tags=["Visualization"], summary="Create Report Chart")
async def generate_report_image(req: ReportChartRequest):
    img = await run_in_threadpool(report_service.generate_report_image, req.logs, req.member_name, req.start_date,
                                  req.end_date)
    return {"status": "success", "image_base64": img}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)