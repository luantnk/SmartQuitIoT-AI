import os
import torch
import onnxruntime as rt
from dotenv import load_dotenv
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

# Global Variables
onnx_session_success = None
onnx_session_craving = None

# Transformers Globals
text_classifier = None
image_processor = None
image_model = None
stt_pipeline = None
tts_processor = None
tts_model = None
tts_vocoder = None
speaker_embeddings = None
hf_client = None


def load_onnx_models():
    global onnx_session_success, onnx_session_craving
    print(f"[INFO] Scanning for ONNX models in: {MODELS_DIR}")

    # Paths
    success_path = os.path.join(MODELS_DIR, "smartquit_success_model.onnx")
    craving_path = os.path.join(MODELS_DIR, "smartquit_craving_time_model.onnx")

    # Fallback for old filename
    if not os.path.exists(success_path):
        fallback = os.path.join(MODELS_DIR, "smartquit_model.onnx")
        if os.path.exists(fallback):
            success_path = fallback

    # Load Success Model
    if os.path.exists(success_path):
        try:
            onnx_session_success = rt.InferenceSession(success_path)
            print(f"[SUCCESS] Loaded Success Model: {success_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load Success Model: {e}")
            onnx_session_success = None
    else:
        print(f"[WARNING] Success Model not found at {success_path}")
        onnx_session_success = None

    # Load Craving Model
    if os.path.exists(craving_path):
        try:
            onnx_session_craving = rt.InferenceSession(craving_path)
            print(f"[SUCCESS] Loaded Craving Model: {craving_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load Craving Model: {e}")
            onnx_session_craving = None
    else:
        print(f"[WARNING] Craving Model not found at {craving_path}")


def load_transformer_models():
    """
    Loads heavy Transformer models. Usually only run once at startup.
    """
    global text_classifier, image_processor, image_model, stt_pipeline
    global tts_processor, tts_model, tts_vocoder, speaker_embeddings, hf_client

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Transformer models on device: {device}")

    try:
        # Toxic Text
        text_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
        )
        # NSFW Image
        image_processor = AutoImageProcessor.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        image_model = AutoModelForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection"
        )
        # Speech to Text
        stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
        )
        # Text to Speech
        tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Embedding
        emb_path = os.path.join(MODELS_DIR, "speaker_speecht5.pt")
        if os.path.exists(emb_path):
            speaker_embeddings = torch.load(emb_path)
        else:
            local_emb = os.path.join(APP_DIR, "speaker_speecht5.pt")
            if os.path.exists(local_emb):
                speaker_embeddings = torch.load(local_emb)
            else:
                print("[WARNING] speaker_speecht5.pt not found.")

        print("[SUCCESS] Transformer models loaded.")
    except Exception as e:
        print(f"[ERROR] Transformer load failed: {e}")

    # Hugging Face Client
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            hf_client = InferenceClient(token=token)
            print("[SUCCESS] Hugging Face Client Connected.")
        except Exception as e:
            print(f"[ERROR] Hugging Face Client failed: {e}")


load_onnx_models()
load_transformer_models()
