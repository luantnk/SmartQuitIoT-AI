import cv2
import numpy as np
import requests
from PIL import Image
import io
import os
import tempfile
from app.models import text_classifier, image_processor, image_model



def is_text_toxic(text: str) -> bool:
    if not text:
        return False
    results = text_classifier(text)

    for item in results[0]:
        if item['label'] != 'neutral' and item['score'] > 0.7:
            return True
    return False



def is_image_nsfw(pil_image) -> bool:
    inputs = image_processor(images=pil_image, return_tensors="pt")
    outputs = image_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = image_model.config.id2label[predicted_class_idx]
    return label == "nsfw"



def check_image_url(url: str) -> bool:
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return is_image_nsfw(image)
    except Exception as e:
        print(f"Error checking image: {e}")
        return False



def check_video_url(url: str) -> bool:
    temp_file = None
    cap = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        temp_file = os.fdopen(temp_fd, "wb")
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            temp_file.write(chunk)
        temp_file.close()


        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        is_nsfw_found = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break


            if frame_count % 30 == 0:

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                if is_image_nsfw(pil_image):
                    is_nsfw_found = True
                    break

            frame_count += 1

        return is_nsfw_found

    except Exception as e:
        print(f"Error checking video: {e}")
        return False
    finally:
        if cap: cap.release()
        if temp_file: os.remove(temp_path)