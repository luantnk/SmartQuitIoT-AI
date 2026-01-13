import os
import torch
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)

print("Loading AI Models... This may take a while...")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


text_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
image_processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
image_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")


stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


print("Loading Speaker Embeddings from file...")


embedding_path = os.path.join(CURRENT_DIR, "models", "speaker_speecht5.pt")

speaker_embeddings = None

if os.path.exists(embedding_path):
    try:

        speaker_embeddings = torch.load(embedding_path)
        print(f"SUCCESS: Speaker Embeddings Loaded from: {embedding_path}")
    except Exception as e:
        print(f"ERROR: Failed to load .pt file: {e}")
else:
    print(f"WARNING: File not found at: {embedding_path}")
    print("ACTION REQUIRED: Please run 'python app/create_embedding.py' to generate it.")

print("All AI Models Ready!")