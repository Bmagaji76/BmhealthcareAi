import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import speech_recognition as sr
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and tokenizer
MODEL_PATH = "model.safetensors"
TOKENIZER_PATH = "tokenizer.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

@app.route("/")
def home():
    return "BM Healthcare AI is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if "text" in data:
            input_text = data["text"]
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({"response": prediction})

        elif "audio" in request.files:
            audio_file = request.files["audio"]
            audio = AudioSegment.from_file(audio_file)
            audio.export("temp.wav", format="wav")

            recognizer = sr.Recognizer()
            with sr.AudioFile("temp.wav") as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)

            return jsonify({"recognized_text": text})

        else:
            return jsonify({"error": "Provide 'text' or 'audio'."})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
