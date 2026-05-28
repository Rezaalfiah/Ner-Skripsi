import os
import re
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request
from markupsafe import escape
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flask init
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

MAX_ABSTRACT_CHARS = 10_000
MAX_TOKENS = 1_000
LSTM_MAX_LEN = 150

# ===== Load Model & Mapping =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "models")

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

def tokenize(text, lowercase=True):
    """Tokenisasi sederhana (jaga tanda baca tetap ada)"""
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens

def word_to_features(tokens, index):
    word = tokens[index]
    features = {
        "word.lower()": word.lower(),
        "word.istitle()": word.istitle(),
        "word.isupper()": word.isupper(),
    }

    if index > 0:
        prev_word = tokens[index - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True

    if index < len(tokens) - 1:
        next_word = tokens[index + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True

    return features

def predict_lstm_tokens(tokens, max_len=LSTM_MAX_LEN):
    if not tokens:
        return []

    seq = [word2idx.get(w, word2idx.get("UNK", 0)) for w in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding='post', truncating='post')

    pred = lstm_model.predict(padded, verbose=0)[0]
    tags = [idx2tag.get(int(np.argmax(p)), "O") for p in pred[:len(tokens)]]
    return list(zip(tokens, tags))

def predict_lstm_sentence(sentence, max_len=LSTM_MAX_LEN):
    tokens = tokenize(sentence)
    predictions = []

    for start in range(0, len(tokens), max_len):
        predictions.extend(predict_lstm_tokens(tokens[start:start + max_len], max_len))

    return predictions

def predict_lstm(text, max_len=LSTM_MAX_LEN, max_tokens=MAX_TOKENS):
    sentences = split_sentences(text)
    all_preds = []
    remaining_tokens = max_tokens

    for s in sentences:
        if remaining_tokens <= 0:
            break

        tokens = tokenize(s)[:remaining_tokens]
        for start in range(0, len(tokens), max_len):
            all_preds.extend(predict_lstm_tokens(tokens[start:start + max_len], max_len))

        remaining_tokens -= len(tokens)

    return all_preds

def predict_nb(text, max_tokens=MAX_TOKENS):
    tokens = tokenize(text, lowercase=False)[:max_tokens]
    if not tokens:
        return []

    features = [word_to_features(tokens, index) for index in range(len(tokens))]
    tags = nb_model.predict(features)
    return list(zip(tokens, tags))

def generate_output(predicted):
    output_parts = []
    herb, body, disease = set(), set(), set()

    for word, tag in predicted:
        clean_word = word.strip('.,')
        escaped_word = escape(word)
        tag_text = str(tag)
        normalized_tag = tag_text.split('-')[-1] if '-' in tag_text else tag_text

        if normalized_tag == 'HERB':
            output_parts.append(f"<span class='highlight-HERB'>{escaped_word}</span>")
            if clean_word:
                herb.add(clean_word)
        elif normalized_tag == 'BODY_PART':
            output_parts.append(f"<span class='highlight-BODY_PART'>{escaped_word}</span>")
            if clean_word:
                body.add(clean_word)
        elif normalized_tag == 'DISEASE':
            output_parts.append(f"<span class='highlight-DISEASE'>{escaped_word}</span>")
            if clean_word:
                disease.add(clean_word)
        else:
            output_parts.append(str(escaped_word))

    return ' '.join(output_parts), ', '.join(sorted(herb)), ', '.join(sorted(body)), ', '.join(sorted(disease))

# ===== Flask Routes =====
@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    output_text = ""
    herbs = body_parts = diseases = ""
    judul = penulis = tahun = ""
    timestamp = f"{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}"

    if request.method == 'POST':
        input_text = request.form.get('abstrak', '')[:MAX_ABSTRACT_CHARS]
        model_choice = request.form.get('model', 'lstm')
        judul = request.form.get('judul', '')
        penulis = request.form.get('penulis', '')
        tahun = request.form.get('tahun', '')

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
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_mode)
