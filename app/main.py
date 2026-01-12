from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.content_moderation import is_text_toxic, check_image_url, check_video_url

app = FastAPI(title="SmartQuitIoT AI Service")


class TextCheckRequest(BaseModel):
    text: str

class MediaUrlRequest(BaseModel):
    url: str

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "AI Service is ready"}

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
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)