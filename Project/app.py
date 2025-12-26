import os
import re
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flask init
app = Flask(__name__)

# ===== Load Model & Mapping =====
model_dir = "models"

# Load LSTM
lstm_model = load_model(os.path.join(model_dir, "ner_lstm_model.h5"))
with open(os.path.join(model_dir, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)
with open(os.path.join(model_dir, "tag2idx.pkl"), "rb") as f:
    tag2idx = pickle.load(f)
idx2tag = {i: t for t, i in tag2idx.items()}

# Load Naive Bayes (pipeline vectorizer + classifier)
with open(os.path.join(model_dir, "ner_naive_bayes.pkl"), "rb") as f:
    nb_model = pickle.load(f)

# ===== Utility Functions =====
def split_sentences(text):
    """Pisahkan teks panjang menjadi kalimat"""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize(text):
    """Tokenisasi sederhana (jaga tanda baca tetap ada)"""
    return re.findall(r"\w+|[^\w\s]", text.lower())

def predict_lstm_sentence(sentence, max_len=150):
    tokens = tokenize(sentence)
    seq = [word2idx.get(w, word2idx.get("UNK", 0)) for w in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')

    pred = lstm_model.predict(padded)[0]
    tags = [idx2tag[np.argmax(p)] for p in pred][:len(tokens)]
    return list(zip(tokens, tags))

def predict_lstm(text, max_len=150):
    sentences = split_sentences(text)
    all_preds = []
    for s in sentences:
        all_preds.extend(predict_lstm_sentence(s, max_len))
    return all_preds

def predict_nb(text):
    tokens = tokenize(text)
    preds = []
    for tok in tokens:
        tag = nb_model.predict([tok])[0]  # prediksi per token
        preds.append((tok, tag))
    return preds

def generate_output(predicted):
    output_html = ""
    herb, body, disease = set(), set(), set()

    for word, tag in predicted:
        clean_word = word.strip('.,')
        normalized_tag = tag.split('-')[-1] if '-' in tag else tag

        if normalized_tag == 'HERB':
            output_html += f"<span class='highlight-HERB'>{word}</span> "
            herb.add(clean_word)
        elif normalized_tag == 'BODY_PART':
            output_html += f"<span class='highlight-BODY_PART'>{word}</span> "
            body.add(clean_word)
        elif normalized_tag == 'DISEASE':
            output_html += f"<span class='highlight-DISEASE'>{word}</span> "
            disease.add(clean_word)
        else:
            output_html += word + " "

    return output_html.strip(), ', '.join(herb), ', '.join(body), ', '.join(disease)

# ===== Flask Routes =====
@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    output_text = ""
    herbs = body_parts = diseases = ""
    judul = penulis = tahun = ""
    timestamp = f"{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}"

    if request.method == 'POST':
        input_text = request.form['abstrak']
        model_choice = request.form['model']
        judul = request.form['judul']
        penulis = request.form['penulis']
        tahun = request.form['tahun']

        if model_choice == 'lstm':
            predicted = predict_lstm(input_text)
        elif model_choice == 'naive_bayes':
            predicted = predict_nb(input_text)
        else:
            predicted = []

        output_text, herbs, body_parts, diseases = generate_output(predicted)

    return render_template('index.html',
                           output_text=output_text,
                           herbs=herbs,
                           body_parts=body_parts,
                           diseases=diseases,
                           timestamp=timestamp,
                           judul=judul,
                           penulis=penulis,
                           tahun=tahun)

if __name__ == '__main__':
    app.run(debug=True)
