import soundfile as sf
from app.models import (
    stt_pipeline,
    tts_processor,
    tts_model,
    tts_vocoder,
    speaker_embeddings,
)


def transcribe_audio_file(file_path: str) -> str:

    try:
        result = stt_pipeline(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error STT: {e}")
        raise e


def text_to_speech_file(text: str, output_path: str):

    if speaker_embeddings is None:
        raise RuntimeError("Speaker embeddings failed to load, cannot generate speech.")

    try:
        inputs = tts_processor(text=text, return_tensors="pt")
        speech = tts_model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=tts_vocoder
        )
        sf.write(output_path, speech.numpy(), samplerate=16000)
        return output_path
    except Exception as e:
        print(f"Error TTS: {e}")
        raise e
