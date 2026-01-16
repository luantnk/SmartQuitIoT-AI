import os
import torch
import onnxruntime as rt
from dotenv import load_dotenv

# HuggingFace / Transformers Imports
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from huggingface_hub import InferenceClient

load_dotenv()

print("[INFO] AI Service Startup: Initializing Models...")

# Path Configuration
CURRENT_FILE_PATH = os.path.abspath(__file__)
APP_DIR = os.path.dirname(CURRENT_FILE_PATH)
MODELS_DIR = os.path.join(APP_DIR, "models")

print(f"[INFO] Scanning for ONNX models in: {MODELS_DIR}")

# Load ONNX Models (Success Prediction & Craving Time)
onnx_session_success = None
onnx_session_craving = None

# Define expected filenames
success_model_path = os.path.join(MODELS_DIR, "smartquit_success_model.onnx")
craving_model_path = os.path.join(MODELS_DIR, "smartquit_craving_time_model.onnx")

# Fallback for the original filename if the new one doesn't exist
if not os.path.exists(success_model_path):
    fallback_path = os.path.join(MODELS_DIR, "smartquit_model.onnx")
    if os.path.exists(fallback_path):
        success_model_path = fallback_path

# Load Success Model
if os.path.exists(success_model_path):
    try:
        onnx_session_success = rt.InferenceSession(success_model_path)
        print(f"[SUCCESS] Loaded Success Prediction Model: {success_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load Success Model: {e}")
else:
    print(f"[WARNING] Success Model not found at {success_model_path}")

# Load Peak Craving Model
if os.path.exists(craving_model_path):
    try:
        onnx_session_craving = rt.InferenceSession(craving_model_path)
        print(f"[SUCCESS] Loaded Peak Craving Model: {craving_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load Craving Model: {e}")
else:
    print(f"[WARNING] Craving Model not found at {craving_model_path}")

# Load Transformers (Content Moderation & Audio)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading Transformer models on device: {device}")

text_classifier = None
image_processor = None
image_model = None
stt_pipeline = None
tts_processor = None
tts_model = None
tts_vocoder = None
speaker_embeddings = None

try:
    # Toxic Text Detection
    text_classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )

    # NSFW Image Detection
    image_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
    image_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

    # Speech to Text (Whisper)
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Text to Speech (SpeechT5)
    tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Load Speaker Embedding
    embedding_path = os.path.join(MODELS_DIR, "speaker_speecht5.pt")
    if os.path.exists(embedding_path):
        speaker_embeddings = torch.load(embedding_path)
    else:
        # Fallback to current dir if not in models folder
        embedding_path_local = os.path.join(APP_DIR, "speaker_speecht5.pt")
        if os.path.exists(embedding_path_local):
            speaker_embeddings = torch.load(embedding_path_local)
        else:
            print("[WARNING] speaker_speecht5.pt not found. TTS might fail.")

    print("[SUCCESS] All Transformer models loaded.")

except Exception as e:
    print(f"[ERROR] Error loading Transformer models: {e}")

# Hugging Face Inference Client
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf_client = None

if hf_token:
    try:
        hf_client = InferenceClient(token=hf_token)
        print("[SUCCESS] Hugging Face Client Connected.")
    except Exception as e:
        print(f"[ERROR] Hugging Face Client connection failed: {e}")
else:
    print("[WARNING] HUGGINGFACEHUB_API_TOKEN not found in env.")

print("[INFO] AI Models Initialization Complete.")