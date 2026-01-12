from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import requests
from app.services.content_moderation import is_text_toxic, check_image_url, check_video_url

app = FastAPI(
    title="SmartQuitIoT AI Service",
    description="Microservice for detecting toxic text and NSFW images/videos.",
    version="1.0.0"
)

class TextCheckRequest(BaseModel):
    text: str

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "AI Service is ready"}

@app.post("/check-content", tags=["Content Moderation"])
def api_check_text(req: TextCheckRequest):
    toxic = is_text_toxic(req.text)
    return {
        "isToxic": toxic,
        "type": "text",
        "message": "Content flagged" if toxic else "Content safe"
    }

@app.post("/check-image-url", tags=["Content Moderation"])
def api_check_image(image_url: str = Form(...)):
    try:
        nsfw = check_image_url(image_url)
        return {"isToxic": nsfw, "type": "image"}

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Could not download image (404 or Network Error)")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid Image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")

@app.post("/check-video-url", tags=["Content Moderation"])
def api_check_video(video_url: str = Form(...)):
    try:
        nsfw = check_video_url(video_url)
        return {"isToxic": nsfw, "type": "video"}

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Could not download video (404 or Network Error)")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid Video: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)