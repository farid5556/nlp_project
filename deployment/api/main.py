from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Inisialisasi FastAPI
app = FastAPI()

# Load model dan tokenizer
MODEL_PATH = r"D:\Pemrograman\python\nlp_project\training\saved_models\classification_sentiment_model"
model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# Struktur data input
class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input_data: TextInput):
    # Tokenisasi teks input
    inputs = tokenizer(
        input_data.text, return_tensors="tf", padding=True, truncation=True, max_length=512
    )

    # Lakukan prediksi
    output = model(**inputs)

    # Ambil hasil softmax
    prediction = tf.nn.softmax(output.logits).numpy()[0]

    # Label kelas (sesuai model training)
    labels = ["negative", "positive"]

    # Ambil kelas dengan probabilitas tertinggi
    predicted_class = labels[prediction.argmax()]
    confidence = float(prediction.max())

    # Kembalikan hasil prediksi
    return {"prediction": predicted_class, "confidence": confidence}

# Untuk running = uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# http://127.0.0.1:8000/predict
# exec bash = Untuk menghentikan .venv di terminal