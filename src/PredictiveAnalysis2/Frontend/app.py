"""
Streamlit app to extract entities from appeal letters (PDF, DOCX, TXT)
using a pretrained Named Entity Recognition (NER) model.
"""

import re
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PyPDF2 import PdfReader
from docx import Document
import os

# Load model and utilities
backend_path = os.path.join(os.path.dirname(__file__), "../Backend")

model = load_model(os.path.join(backend_path, "ner_model.h5"))

with open(os.path.join(backend_path, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(backend_path, "idx_to_label.pkl"), "rb") as f:
    idx_to_label = pickle.load(f)

with open(os.path.join(backend_path, "max_len.pkl"), "rb") as f:
    max_len = pickle.load(f)

word_index = tokenizer.word_index


def predict_entities(text):
    """
    Predicts entities from the input text using the trained model.
    Args:
        text (str): The input text to analyze.
    Returns:
        dict: A dictionary containing the extracted entities.
    """
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


def clean_entity(text):
    """
    Cleans unwanted characters like punctuation from the entity text.
    Args:
        text (str): The text to clean.
    Returns:
        str: The cleaned text.
    """
    return re.sub(r'^[^\w]*|[^\w]*$', '', text).strip()


def read_pdf(file):
    """
    Reads and extracts text from a PDF file.
    Args:
        file (file): The PDF file to read.
    Returns:
        str: Extracted text from the PDF.
    """
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages)


def read_docx(file):
    """
    Reads and extracts text from a DOCX file.
    Args:
        file (file): The DOCX file to read.
    Returns:
        str: Extracted text from the DOCX file.
    """
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def read_txt(file):
    """
    Reads and extracts text from a TXT file.
    Args:
        file (file): The TXT file to read.
    Returns:
        str: Extracted text from the TXT file.
    """
    return file.read().decode("utf-8")


# Streamlit UI
st.title("Appeal Letter Entity Extractor")

uploaded_file = st.file_uploader(
    "Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
)

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        EXTRACTED_TEXT = read_pdf(uploaded_file)
    elif uploaded_file.type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        EXTRACTED_TEXT = read_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        EXTRACTED_TEXT = read_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        EXTRACTED_TEXT = ""

    if EXTRACTED_TEXT:
        st.subheader("Extracted Text:")
        st.write(EXTRACTED_TEXT)

        result = predict_entities(EXTRACTED_TEXT)

        st.subheader("Extracted Entities:")
        st.write(f"**Claim Number:** {result['claim_number']}")
        st.write(f"**Reason for Denial:** {result['reason_for_denial']}")
        st.write(f"**Doctor Name:** {result['doctor_name']}")
        st.write(
            f"**Health Plan Name:** {result['health_plan_name']}"
        )
