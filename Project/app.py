import os
import re
import pickle
import numpy as np
from datetime import datetime

import tensorflow as tf
from flask import Flask, render_template, request
from markupsafe import escape


# =========================================================
# Flask Config
# =========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

MAX_ABSTRACT_CHARS = 10_000
MAX_TOKENS = 1_000


# =========================================================
# Model Paths
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MAPPING_PATH = os.path.join(MODEL_DIR, "ner_mappings.pkl")
BILSTM_WEIGHTS_PATH = os.path.join(MODEL_DIR, "bilstm_ner_weights.weights.h5")
NB_MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_ner_model.pkl")


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            f"Pastikan file tersebut berada di folder models."
        )


require_file(MAPPING_PATH)
require_file(BILSTM_WEIGHTS_PATH)
require_file(NB_MODEL_PATH)


# =========================================================
# Load Mapping
# =========================================================

with open(MAPPING_PATH, "rb") as f:
    mappings = pickle.load(f)

word2idx = mappings["word2idx"]
tag2idx = mappings["tag2idx"]
idx2tag = mappings["idx2tag"]

# Pastikan idx2tag key bertipe integer
idx2tag = {int(k): v for k, v in idx2tag.items()}

MAX_LEN = int(mappings.get("max_len", 150))
BEST_THRESHOLD = float(mappings.get("best_threshold", 0.45))
LOWER_TOKEN = bool(mappings.get("lower_token", True))

PAD_TAG = mappings.get("pad_tag", "<PAD>")
UNK_TOKEN = mappings.get("unk_token", "<UNK>")

UNK_ID = word2idx.get(
    UNK_TOKEN,
    word2idx.get("<UNK>", word2idx.get("UNK", 1))
)

VOCAB_SIZE = len(word2idx)
TAG_SIZE = len(tag2idx)


# =========================================================
# Build BiLSTM Architecture
# =========================================================

def build_bilstm_model(vocab_size, tag_size, max_len):
    inputs = tf.keras.layers.Input(
        shape=(max_len,),
        dtype="int32",
        name="input_ids"
    )

    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=128,
        mask_zero=True,
        name="embedding"
    )(inputs)

    x = tf.keras.layers.SpatialDropout1D(0.30)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.30,
            recurrent_dropout=0.0
        )
    )(x)

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(64, activation="relu")
    )(x)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(tag_size, activation="softmax")
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Load BiLSTM weights only
bilstm_model = build_bilstm_model(
    vocab_size=VOCAB_SIZE,
    tag_size=TAG_SIZE,
    max_len=MAX_LEN
)

bilstm_model.load_weights(BILSTM_WEIGHTS_PATH)


# =========================================================
# Load Naive Bayes Model
# =========================================================

with open(NB_MODEL_PATH, "rb") as f:
    nb_model = pickle.load(f)


# =========================================================
# Model Information for Template
# =========================================================

MODEL_INFO = {
    "bilstm": {
        "name": "BiLSTM",
        "output_title": "Output BiLSTM",
        "description": (
            "Model BiLSTM digunakan untuk mengenali entitas berdasarkan urutan token. "
            "Model ini membaca konteks dari token sebelum dan sesudahnya, sehingga lebih sesuai "
            "untuk tugas Named Entity Recognition berbasis BIO tagging."
        ),
        "role": "Model utama",
        "score": "F1-score: 0.6165"
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "output_title": "Output Naive Bayes",
        "description": (
            "Model Naive Bayes digunakan sebagai model pembanding atau baseline. "
            "Model ini melakukan klasifikasi berdasarkan fitur token, tetapi tidak sekuat BiLSTM "
            "dalam memahami hubungan sekuensial antar-token."
        ),
        "role": "Model pembanding",
        "score": "F1-score: 0.3907"
    }
}


# =========================================================
# Text Cleaning & Tokenization
# =========================================================

def clean_text_for_ner(text):
    text = str(text)

    text = text.replace("\n", " ")
    text = re.sub(r"\.(?=\w)", ". ", text)

    text = re.sub(r"\banti\s+inflamasi\b", "antiinflamasi", text, flags=re.IGNORECASE)
    text = re.sub(r"\banti\s+bakteri\b", "antibakteri", text, flags=re.IGNORECASE)
    text = re.sub(r"\banti\s+oksidan\b", "antioksidan", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    text = clean_text_for_ner(text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def normalize_token(token):
    token = str(token)
    return token.lower() if LOWER_TOKEN else token


# =========================================================
# BIO Utilities
# =========================================================

def fix_bio_sequence(tags):
    fixed = []
    prev = "O"

    for tag in tags:
        tag = str(tag)

        if tag.startswith("I-"):
            entity = tag[2:]
            valid_prev = {f"B-{entity}", f"I-{entity}"}

            if prev not in valid_prev:
                tag = f"B-{entity}"

        fixed.append(tag)
        prev = tag

    return fixed


def normalize_entity_label(tag):
    tag = str(tag)

    if tag == "O":
        return "O"

    if "-" in tag:
        return tag.split("-", 1)[1]

    return tag


# =========================================================
# BiLSTM Prediction
# =========================================================

LABEL_THRESHOLDS = {
    "B-EFFECT": 0.60,
    "I-EFFECT": 0.60,

    "B-TREATMENT": 0.52,
    "I-TREATMENT": 0.52,

    "B-METHOD": 0.35,
    "I-METHOD": 0.35,

    "B-POPULATION": 0.40,
    "I-POPULATION": 0.40,
}


def predict_bilstm_tokens(tokens):
    if not tokens:
        return []

    chunk_tokens = tokens[:MAX_LEN]
    normalized_tokens = [normalize_token(t) for t in chunk_tokens]

    x = np.zeros((1, MAX_LEN), dtype=np.int32)

    for i, token in enumerate(normalized_tokens):
        x[0, i] = word2idx.get(token, UNK_ID)

    pred_proba = bilstm_model.predict(x, verbose=0)[0]
    pred_ids = np.argmax(pred_proba, axis=-1)
    pred_conf = np.max(pred_proba, axis=-1)

    tags = []

    for i in range(len(chunk_tokens)):
        tag = idx2tag.get(int(pred_ids[i]), "O")

        if tag == PAD_TAG:
            tag = "O"

        if tag != "O":
            threshold = LABEL_THRESHOLDS.get(tag, BEST_THRESHOLD)

            if pred_conf[i] < threshold:
                tag = "O"

        tags.append(tag)

    tags = fix_bio_sequence(tags)

    return list(zip(chunk_tokens, tags))


def predict_bilstm(text, max_tokens=MAX_TOKENS):
    tokens = tokenize(text)[:max_tokens]

    if not tokens:
        return []

    all_predictions = []

    for start in range(0, len(tokens), MAX_LEN):
        chunk = tokens[start:start + MAX_LEN]
        all_predictions.extend(predict_bilstm_tokens(chunk))

    all_tokens = [token for token, _ in all_predictions]
    all_tags = [tag for _, tag in all_predictions]

    all_tags = fix_bio_sequence(all_tags)

    return list(zip(all_tokens, all_tags))


# =========================================================
# Naive Bayes Prediction
# =========================================================

def token_shape(token):
    token = str(token)

    if token.isdigit():
        return "digit"
    if token.isupper():
        return "upper"
    if token.istitle():
        return "title"
    if token.islower():
        return "lower"

    return "mixed"


def word_to_features(tokens, index):
    word = str(tokens[index])

    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-1:]": word[-1:],
        "word[-2:]": word[-2:],
        "word[-3:]": word[-3:],
        "word[:1]": word[:1],
        "word[:2]": word[:2],
        "word.isdigit": word.isdigit(),
        "word.shape": token_shape(word),
        "position": index,
        "relative_position": round(index / max(len(tokens), 1), 2),
    }

    if index > 0:
        prev_word = str(tokens[index - 1])

        features.update({
            "prev.word.lower": prev_word.lower(),
            "prev.word[-3:]": prev_word[-3:],
            "prev.word.shape": token_shape(prev_word),
        })
    else:
        features["BOS"] = True

    if index < len(tokens) - 1:
        next_word = str(tokens[index + 1])

        features.update({
            "next.word.lower": next_word.lower(),
            "next.word[-3:]": next_word[-3:],
            "next.word.shape": token_shape(next_word),
        })
    else:
        features["EOS"] = True

    return features


def sent2features(tokens):
    return [word_to_features(tokens, i) for i in range(len(tokens))]


def predict_nb(text, max_tokens=MAX_TOKENS):
    original_tokens = tokenize(text)[:max_tokens]

    if not original_tokens:
        return []

    normalized_tokens = [normalize_token(t) for t in original_tokens]
    features = sent2features(normalized_tokens)

    tags = nb_model.predict(features).tolist()
    tags = fix_bio_sequence(tags)

    return list(zip(original_tokens, tags))


# =========================================================
# Output Formatting
# =========================================================

ENTITY_LABELS = [
    "HERB",
    "BODY_PART",
    "DISEASE",
    "COMPOUND",
    "EFFECT",
    "TREATMENT",
    "METHOD",
    "POPULATION",
]


def extract_entities(predicted):
    entities = []
    current_tokens = []
    current_label = None

    for token, tag in predicted:
        tag = str(tag)

        if tag == "O":
            if current_tokens:
                entities.append({
                    "entity": " ".join(current_tokens),
                    "label": current_label,
                })

                current_tokens = []
                current_label = None

            continue

        if "-" not in tag:
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B":
            if current_tokens:
                entities.append({
                    "entity": " ".join(current_tokens),
                    "label": current_label,
                })

            current_tokens = [token]
            current_label = label

        elif prefix == "I":
            if current_tokens and current_label == label:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "entity": " ".join(current_tokens),
                        "label": current_label,
                    })

                current_tokens = [token]
                current_label = label

    if current_tokens:
        entities.append({
            "entity": " ".join(current_tokens),
            "label": current_label,
        })

    return entities


def detokenize_html(parts):
    text = " ".join(parts)

    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def generate_output(predicted):
    output_parts = []
    entity_sets = {label: set() for label in ENTITY_LABELS}

    for word, tag in predicted:
        escaped_word = escape(word)
        label = normalize_entity_label(tag)

        if label in ENTITY_LABELS:
            output_parts.append(
                f"<span class='highlight-{label}'>{escaped_word}</span>"
            )
        else:
            output_parts.append(str(escaped_word))

    entities = extract_entities(predicted)

    for item in entities:
        label = item["label"]
        entity = item["entity"].strip(" .,;:!?()[]{}")

        if label in entity_sets and entity:
            entity_sets[label].add(entity)

    highlighted_text = detokenize_html(output_parts)

    return {
        "output_text": highlighted_text,
        "herbs": ", ".join(sorted(entity_sets["HERB"])),
        "body_parts": ", ".join(sorted(entity_sets["BODY_PART"])),
        "diseases": ", ".join(sorted(entity_sets["DISEASE"])),
        "compounds": ", ".join(sorted(entity_sets["COMPOUND"])),
        "effects": ", ".join(sorted(entity_sets["EFFECT"])),
        "treatments": ", ".join(sorted(entity_sets["TREATMENT"])),
        "methods": ", ".join(sorted(entity_sets["METHOD"])),
        "populations": ", ".join(sorted(entity_sets["POPULATION"])),
        "entities": entities,
    }


# =========================================================
# Flask Routes
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""

    result = {
        "output_text": "",
        "herbs": "",
        "body_parts": "",
        "diseases": "",
        "compounds": "",
        "effects": "",
        "treatments": "",
        "methods": "",
        "populations": "",
        "entities": [],
    }

    judul = ""
    penulis = ""
    tahun = ""
    model_choice = "bilstm"

    timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    if request.method == "POST":
        input_text = request.form.get("abstrak", "")[:MAX_ABSTRACT_CHARS]
        raw_model_choice = request.form.get("model", "bilstm")

        judul = request.form.get("judul", "")
        penulis = request.form.get("penulis", "")
        tahun = request.form.get("tahun", "")

        if raw_model_choice in {"bilstm", "lstm"}:
            model_choice = "bilstm"
            predicted = predict_bilstm(input_text)

        elif raw_model_choice in {"naive_bayes", "nb"}:
            model_choice = "naive_bayes"
            predicted = predict_nb(input_text)

        else:
            model_choice = "bilstm"
            predicted = []

        result = generate_output(predicted)

    model_name = MODEL_INFO[model_choice]["name"]
    model_output_title = MODEL_INFO[model_choice]["output_title"]
    model_description = MODEL_INFO[model_choice]["description"]
    model_role = MODEL_INFO[model_choice]["role"]
    model_score = MODEL_INFO[model_choice]["score"]

    return render_template(
        "index.html",
        input_text=input_text,
        output_text=result["output_text"],
        herbs=result["herbs"],
        body_parts=result["body_parts"],
        diseases=result["diseases"],
        compounds=result["compounds"],
        effects=result["effects"],
        treatments=result["treatments"],
        methods=result["methods"],
        populations=result["populations"],
        entities=result["entities"],
        timestamp=timestamp,
        judul=judul,
        penulis=penulis,
        tahun=tahun,
        model_choice=model_choice,
        model_name=model_name,
        model_output_title=model_output_title,
        model_description=model_description,
        model_role=model_role,
        model_score=model_score,
        best_threshold=BEST_THRESHOLD,
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_mode)