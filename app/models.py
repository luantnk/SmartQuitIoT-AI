# app/models.py
import os
import torch
import onnxruntime as rt
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


load_dotenv()

print("Loading AI Models... This may take a while...")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading Content Moderation & Audio Models...")
try:
    # Toxic Comment Detection
    text_classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )

    # NSFW Image Detection
    image_processor = AutoImageProcessor.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )
    image_model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection"
    )

    # Speech to Text (Whisper)
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Text to Speech (Microsoft SpeechT5)
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Load Speaker Embeddings for TTS
    embedding_path = os.path.join(CURRENT_DIR, "speaker_speecht5.pt")
    speaker_embeddings = None
    if os.path.exists(embedding_path):
        speaker_embeddings = torch.load(embedding_path)
    else:
        print("WARNING: speaker_speecht5.pt not found. TTS might fail.")

except Exception as e:
    print(f"ERROR loading local models: {e}")


print("Initializing Hugging Face Inference Client...")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print(
        "WARNING: HUGGINGFACEHUB_API_TOKEN not found in env. Summary service will fail."
    )
    hf_client = None
else:
    hf_client = InferenceClient(token=hf_token)
    print("SUCCESS: Hugging Face Client Connected!")


print("Loading ONNX Prediction Model...")
candidate_paths = [
    os.path.join(CURRENT_DIR, "smartquit_model.onnx"),
    os.path.join(BASE_DIR, "models", "smartquit_model.onnx"),
    "smartquit_model.onnx",
]

onnx_session = None
model_path = None

for path in candidate_paths:
    if os.path.exists(path):
        try:
            onnx_session = rt.InferenceSession(path)
            model_path = path
            print(f"SUCCESS: ONNX Model loaded from: {path}")
            break
        except Exception as e:
            print(f"Found file but failed to load at {path}: {e}")

if onnx_session is None:
    print("WARNING: Could not find smartquit_model.onnx")

print("All AI Models Ready!")
