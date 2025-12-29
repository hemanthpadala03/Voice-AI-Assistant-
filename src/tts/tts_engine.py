import pyttsx3
import os

class TTSEngine:
    def __init__(self, rate=170):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def speak_to_file(self, text, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
