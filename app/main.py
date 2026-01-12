from fastapi import FastAPI, Form
from pydantic import BaseModel
from app.services.content_moderation import is_text_toxic, check_image_url, check_video_url

app = FastAPI(title="SmartQuitIoT AI Service")


class TextCheckRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "AI Service is running"}

# API Check Text
@app.post("/check-content")
def api_check_text(req: TextCheckRequest):
    toxic = is_text_toxic(req.text)
    return {"isToxic": toxic, "type": "text"}

# API Check Image (URL)
@app.post("/check-image-url")
def api_check_image(image_url: str = Form(...)):
    nsfw = check_image_url(image_url)
    return {"isToxic": nsfw, "type": "image"}

# API Check Video (URL)
@app.post("/check-video-url")
def api_check_video(video_url: str = Form(...)):
    nsfw = check_video_url(video_url)
    return {"isToxic": nsfw, "type": "video"}