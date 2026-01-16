import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
import io
import os
import tempfile
from app.models import text_classifier, image_processor, image_model

TOXIC_THRESHOLD = 0.7
VIDEO_FRAME_SKIP = 30


def is_text_toxic(text: str) -> bool:
    if not text:
        return False

    results = text_classifier(text)

    for item in results[0]:
        if item["label"] != "neutral" and item["score"] > TOXIC_THRESHOLD:
            return True
    return False


def is_image_nsfw(pil_image) -> bool:

    inputs = image_processor(images=pil_image, return_tensors="pt")

    outputs = image_model(**inputs)

    predicted_class_idx = outputs.logits.argmax(-1).item()

    label = image_model.config.id2label[predicted_class_idx]

    return label == "nsfw"


def check_image_url(url: str) -> bool:
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    try:
        image = Image.open(io.BytesIO(response.content))
    except UnidentifiedImageError:
        raise ValueError("The file provided is not a valid image.")

    return is_image_nsfw(image)


def check_video_url(url: str) -> bool:
    temp_path = None
    cap = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(temp_fd, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                temp_file.write(chunk)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(
                "Could not open video file. Corrupted or unsupported format."
            )

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:

                if frame_count == 0:
                    raise ValueError(
                        "Video stream is empty or unreadable (moov atom not found)."
                    )
                break

            if frame_count % VIDEO_FRAME_SKIP == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                if is_image_nsfw(pil_image):
                    return True

            frame_count += 1

        return False

    finally:
        if cap:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
