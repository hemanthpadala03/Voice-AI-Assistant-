from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-mul-en"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        translated = self.model.generate(**tokens)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)
