import os
import uvicorn
import shutil
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from app.requests.api_schemas import (
    TextCheckRequest,
    MediaUrlRequest,
    QuitPlanPredictRequest,
    TextToSpeechRequest,
    SummaryRequest,
    DiaryAnalysisRequest,
    ReportChartRequest
)

from app.services.content_moderation_service import is_text_toxic, check_image_url, check_video_url
from app.services.audio_service import transcribe_audio_file, text_to_speech_file

from app.services.summary_service import summary_service, generate_coach_summary
from app.services.report_service import report_service

from app.models import onnx_session, model_path

app = FastAPI(title="SmartQuitIoT AI Service")
CURRENT_WORKING_DIR = os.getcwd()


def cleanup_file(path: str):
    if os.path.exists(path):
        os.remove(path)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "AI Service is ready",
        "onnx_model_loaded": onnx_session is not None,
        "onnx_path": model_path if model_path else "Not Found"
    }


@app.post("/predict-quit-status", tags=["Plan Prediction"])
async def predict_quit_status(req: QuitPlanPredictRequest):
    if onnx_session is None:
        raise HTTPException(status_code=503, detail="Prediction model not found on server")
    try:
        # Logic chạy ONNX
        input_data = np.array([req.features], dtype=np.float32)
        input_name = onnx_session.get_inputs()[0].name
        result = onnx_session.run(None, {input_name: input_data})

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
async def api_check_text(req: TextCheckRequest):
    try:
        toxic = is_text_toxic(req.text)
        return {"isToxic": toxic, "type": "text"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check-image-url", tags=["Content Moderation"])
async def api_check_image(req: MediaUrlRequest):
    try:
        nsfw = check_image_url(req.url)
        return {"isToxic": nsfw, "type": "image"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/check-video-url", tags=["Content Moderation"])
async def api_check_video(req: MediaUrlRequest):
    try:
        nsfw = check_video_url(req.url)
        return {"isToxic": nsfw, "type": "video"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/voice-to-text", tags=["Audio Processing"])
async def api_voice_to_text(file: UploadFile = File(...)):
    filename = file.filename if file.filename else "audio.wav"
    file_extension = filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_extension}"
    temp_file_path = os.path.join(CURRENT_WORKING_DIR, temp_filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        text_result = transcribe_audio_file(temp_file_path)
        return {"text": text_result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/text-to-voice", tags=["Audio Processing"])
async def api_text_to_voice(req: TextToSpeechRequest, background_tasks: BackgroundTasks):
    temp_filename = f"tts_output_{uuid.uuid4()}.wav"
    output_path = os.path.join(CURRENT_WORKING_DIR, temp_filename)
    try:
        text_to_speech_file(req.text, output_path)
        background_tasks.add_task(cleanup_file, output_path)
        return FileResponse(output_path, media_type="audio/wav", filename="speech.wav")
    except Exception as e:
        cleanup_file(output_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-week", tags=["Coach Assistance"])
async def summarize_week(request: SummaryRequest):
    try:
        logs_dict = [log.dict() for log in request.logs]
        summary_text = await run_in_threadpool(generate_coach_summary, request.member_name, logs_dict)
        return {"status": "success", "summary": summary_text}
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-diary", tags=["Coach Assistance"])
async def analyze_diary(request: DiaryAnalysisRequest):
    try:
        data = request.dict()
        result = await run_in_threadpool(summary_service.analyze_diary_sentiment, data)
        return result
    except Exception as e:
        print(f"Daily Analysis Error: {e}")
        return {
            "message": "Keep going!",
            "is_high_risk": False,
            "status_color": "gray"
        }

@app.post("/generate-report-image", tags=["Visualization"])
async def generate_report_image(req: ReportChartRequest):
    try:
        image_base64 = await run_in_threadpool(
            report_service.generate_report_image,
            req.logs,
            req.member_name,
            req.start_date,
            req.end_date
        )

        if not image_base64:
            raise HTTPException(status_code=400, detail="Could not generate image from provided data")

        return {"status": "success", "image_base64": image_base64}

    except Exception as e:
        print(f"Report Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)