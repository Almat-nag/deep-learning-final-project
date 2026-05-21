import re
import pickle
import torch
import torch.nn as nn
import streamlit as st


# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Model architecture
# ============================================================

class LSTMToxicClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        logits = self.fc(hidden)
        return logits


# ============================================================
# Load model and vocabulary
# ============================================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("results/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    PAD_IDX = vocab["<PAD>"]
    vocab_size = len(vocab)

    model = LSTMToxicClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=128,
        output_dim=6,
        pad_idx=PAD_IDX
    )

    model.load_state_dict(
        torch.load("results/final_lstm_model.pt", map_location=device)
    )

    model.to(device)
    model.eval()

    return model, vocab, device


# ============================================================
# Encode text
# ============================================================

def encode_text(text, vocab, max_len=200):
    tokens = text.split()

    ids = []
    for token in tokens:
        ids.append(vocab.get(token, vocab.get("<UNK>", 1)))

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


# ============================================================
# Streamlit application
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Toxic Comment Detection App")

st.write(
    "This application detects toxic and abusive comments using an LSTM deep learning model."
)

st.markdown("---")

user_input = st.text_area(
    "Enter a comment:",
    placeholder="Type a comment here..."
)

threshold = st.slider(
    "Prediction threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.1
)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

if st.button("Analyze Comment"):
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        try:
            model, vocab, device = load_model()

            cleaned_text = clean_text(user_input)
            encoded_text = encode_text(cleaned_text, vocab).to(device)

            with torch.no_grad():
                logits = model(encoded_text)
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]

            st.subheader("Prediction Results")

            predicted_labels = []

            for label, probability in zip(labels, probabilities):
                st.write(f"**{label}:** {probability:.4f}")

                if probability >= threshold:
                    predicted_labels.append(label)

            st.markdown("---")

            if predicted_labels:
                st.error("This comment may be toxic.")
                st.write("Detected categories:")

                for label in predicted_labels:
                    st.write(f"- {label}")
            else:
                st.success("This comment looks non-toxic.")

        except FileNotFoundError:
            st.error(
                "Model files were not found. Please check that final_lstm_model.pt and vocab.pkl are inside the results folder."
            )

        except Exception as e:
            st.error(f"Error: {e}")
