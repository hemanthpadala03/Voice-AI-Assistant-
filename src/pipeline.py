import json
import os
from src.stt.whisper_stt import WhisperSTT
from src.translation.translator import Translator
from src.llm.ollama_client import OllamaClient
from src.tts.tts_engine import TTSEngine

def run_tts():
    tts = TTSEngine()

    for lang in ["en", "hi", "te"]:
        path = f"data/llm_outputs/{lang}_self_01.json"

        if not os.path.exists(path):
            print(f"[SKIP TTS] No LLM output for {lang}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        text = data["response"]

        if not text.strip():
            print(f"[SKIP TTS] Empty response for {lang}")
            continue

        output_audio = f"data/audio_output/{lang}_self_01.wav"
        tts.speak_to_file(text, output_audio)

        print(f"[TTS DONE] {lang}")


def run_llm():
    os.makedirs("data/llm_outputs", exist_ok=True)
    llm = OllamaClient()

    for lang in ["en", "hi", "te"]:
        with open(f"data/translated/{lang}_self_01.json", encoding="utf-8") as f:
            data = json.load(f)

        prompt = data["translated_text"]

        response = llm.generate(prompt)

        with open(f"data/llm_outputs/{lang}_self_01.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": prompt,
                    "response": response
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"[LLM DONE] {lang}")


def run_stt_and_translation():
    stt = WhisperSTT()
    translator = Translator()

    files = {
        "en": "data/audio_input/en_self_01.wav",
        "hi": "data/audio_input/hi_self_01.wav",
        "te": "data/audio_input/te_self_01.wav",
    }

    for lang, path in files.items():
        out = stt.transcribe(path)
        text = out["text"]

        if out["language"] != "en":
            translated = translator.translate(text)
        else:
            translated = text

        with open(f"data/translated/{lang}_self_01.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "original_text": text,
                    "translated_text": translated,
                    "source_language": out["language"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[TRANSLATED] {lang}: {translated}")

if __name__ == "__main__":
    run_stt_and_translation()
    run_llm()
    run_tts()
    

