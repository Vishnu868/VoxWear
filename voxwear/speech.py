"""Speech-to-text (Whisper, 3 s chunks @ 16 kHz) and text-to-speech (pyttsx3)."""
import numpy as np

SAMPLE_RATE = 16000


class SpeechToText:
    def __init__(self, model_size: str = "base", language: str = "en"):
        import whisper
        self.model = whisper.load_model(model_size)
        self.language = language

    def record(self, seconds: float = 3.0) -> np.ndarray:
        import sounddevice as sd
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return audio.ravel()

    def transcribe(self, audio: np.ndarray, silence_rms: float = 0.01):
        if np.sqrt(np.mean(audio ** 2)) < silence_rms:        # silence detection
            return None
        result = self.model.transcribe(audio, language=self.language, fp16=False)
        text = result["text"].strip()
        return text or None


class TextToSpeech:
    def __init__(self, rate: int = 170):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def say(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
