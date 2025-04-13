# Create the Streamlit app

import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PyPDF2 import PdfReader
from docx import Document

# Load model and utilities
model = load_model("ner_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("idx_to_label.pkl", "rb") as f:
    idx_to_label = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

word_index = tokenizer.word_index

# Prediction function
def predict_entities(text):
    words = text.split()
    seq = [word_index.get(w, 1) for w in words]
    seq = pad_sequences([seq], maxlen=max_len, padding='post')

    pred = model.predict(seq)[0]
    pred = np.argmax(pred, axis=-1)

    predicted_labels = [idx_to_label[p] for p in pred[:len(words)]]

    entities = {}
    current_entity = []
    current_type = None

    for i, label in enumerate(predicted_labels):
        if label.startswith("B-"):
            if current_entity:
                entities[current_type] = clean_entity(" ".join(current_entity))
            current_entity = [words[i]]
            current_type = label[2:]
        elif label.startswith("I-") and current_entity and label[2:] == current_type:
            current_entity.append(words[i])
        elif current_entity:
            entities[current_type] = clean_entity(" ".join(current_entity))
            current_entity = []
            current_type = None

    if current_entity:
        entities[current_type] = clean_entity(" ".join(current_entity))

    return {
        "claim_number": entities.get("CLAIM", ""),
        "reason_for_denial": entities.get("REASON", ""),
        "doctor_name": entities.get("DOCTOR", ""),
        "health_plan_name": entities.get("PLAN", "")
    }

# Clean entity function
def clean_entity(text):
    """Cleans unwanted characters like punctuation."""
    return re.sub(r'^[^\w]*|[^\w]*$', '', text).strip()

# File reading functions
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages)

def read_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)

def read_txt(file):
    return file.read().decode("utf-8")

# Streamlit UI
st.title("Appeal Letter Entity Extractor")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        st.subheader("Extracted Text:")
        st.write(text)

        result = predict_entities(text)

        st.subheader("Extracted Entities:")
        st.write(f"**Claim Number:** {result['claim_number']}")
        st.write(f"**Reason for Denial:** {result['reason_for_denial']}")
        st.write(f"**Doctor Name:** {result['doctor_name']}")
        st.write(f"**Health Plan Name:** {result['health_plan_name']}")
