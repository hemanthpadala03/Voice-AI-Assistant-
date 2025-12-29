import requests

class OllamaClient:
    def __init__(self, model="llama2:latest", url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 128   # <<< CRITICAL
            }
        }

        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()
