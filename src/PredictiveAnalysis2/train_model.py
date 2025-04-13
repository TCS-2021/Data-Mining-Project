""" Program that does the model training """

import pickle
import random
import re

import numpy as np
from faker import Faker
from keras.layers import (Bidirectional, Dense, Embedding, Input,
                                     LSTM, TimeDistributed)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

# Initialize Faker
fake = Faker()

# Generate synthetic appeal letters
def generate_appeal_letters(n_samples=500):
    """Generate synthetic appeal letters with entities."""
    claim_numbers = [f"CLM{random.randint(100000, 999999)}" for _ in range(50)]
    denial_reasons = [
        "Service not covered", "Pre-authorization required", "Out of network provider",
        "Insufficient documentation", "Medical necessity not established"
    ]
    health_plans = ["BlueCross BlueShield", "United Healthcare", "Aetna", "Cigna", "Medicare"]

    letters_list, annotations_list = [], []

    for _ in range(n_samples):
        claim_num = random.choice(claim_numbers)
        reason = random.choice(denial_reasons)
        doctor = f"Dr. {fake.name()}"
        plan = random.choice(health_plans)

        appeal_letter = (
            f"Dear Sir/Madam, I am writing to appeal the denial of claim {claim_num}. "
            f'The reason provided was "{reason}". '
            f'The treatment was provided by {doctor} under {plan}. '
            f"Please reconsider this decision. Sincerely, {fake.name()}"
        )

        entities = [(claim_num, "CLAIM"), (reason, "REASON"), (doctor, "DOCTOR"), (plan, "PLAN")]
        letters_list.append(appeal_letter)
        annotations_list.append(entities)

    return letters_list, annotations_list

# Generate dataset
letters_data, annotations_data = generate_appeal_letters(500)

# Tokenization
tokenizer = Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(letters_data)
word_index = tokenizer.word_index
total_vocab_size = len(word_index) + 1

# Label mappings
label_to_idx = {
    "O": 0, "B-CLAIM": 1, "I-CLAIM": 2,
    "B-REASON": 3, "I-REASON": 4,
    "B-DOCTOR": 5, "I-DOCTOR": 6,
    "B-PLAN": 7, "I-PLAN": 8
}
idx_to_label = {i: l for l, i in label_to_idx.items()}

# Preprocessing
X_data, y_data = [], []
max_sequence_len = max(len(text.split()) for text in letters_data)

for text, annotation in zip(letters_data, annotations_data):
    words = text.split()
    x = [word_index.get(w, 1) for w in words]  # 1 = OOV
    labels = ["O"] * len(words)

    for entity, tag in annotation:
        entity_words = entity.split()
        entity_words_cleaned = [re.sub(r"\W+", "", w).lower() for w in entity_words]

        for i in range(len(words) - len(entity_words) + 1):
            window = words[i:i + len(entity_words)]
            window_cleaned = [re.sub(r"\W+", "", w).lower() for w in window]

            if window_cleaned == entity_words_cleaned:
                labels[i] = f"B-{tag}"
                for j in range(1, len(entity_words)):
                    labels[i + j] = f"I-{tag}"

    y_seq = [label_to_idx[l] for l in labels]
    X_data.append(x)
    y_data.append(y_seq)

X_data = pad_sequences(X_data, maxlen=max_sequence_len, padding='post')
y_data = pad_sequences(y_data, maxlen=max_sequence_len, padding='post')
y_data = np.array([to_categorical(i, num_classes=len(label_to_idx)) for i in y_data])

# Build model
def build_bilstm_model(vocab, num_classes, max_len):
    """Build and return a BiLSTM model for NER."""
    inputs = Input(shape=(max_len,))
    embedding = Embedding(input_dim=vocab, output_dim=100)(inputs)
    lstm = Bidirectional(LSTM(
        128,
        return_sequences=True,
        dropout=0.5,
        recurrent_dropout=0.5))(embedding)
    dense = TimeDistributed(Dense(num_classes, activation='softmax'))(lstm)
    model = Model(inputs, dense)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save
model_instance = build_bilstm_model(total_vocab_size, len(label_to_idx), max_sequence_len)
model_instance.fit(
    X_data[:400], y_data[:400],
    validation_data=(X_data[400:], y_data[400:]),
    epochs=10, batch_size=32
)

model_instance.save("ner_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("idx_to_label.pkl", "wb") as f:
    pickle.dump(idx_to_label, f)

with open("max_len.pkl", "wb") as f:
    pickle.dump(max_sequence_len, f)
