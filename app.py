import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import streamlit as st


# Page Config

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide"
)


# Custom CSS

st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }
    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Model Architecture

class LSTMToxicClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden = hidden[-1]
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits


# Preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(text, word2idx, max_len):
    pad_idx = word2idx.get("<PAD>", 0)
    unk_idx = word2idx.get("<UNK>", 1)

    tokens = clean_text(text).split()
    ids = [word2idx.get(token, unk_idx) for token in tokens]

    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return torch.tensor([ids], dtype=torch.long)


# Load Model

@st.cache_resource
def load_model():
    model_path = Path("results/week3_lstm_model.pt")

    if not model_path.exists():
        st.error("Model file not found: results/week3_lstm_model.pt")
        st.info("Place your trained Week 3 LSTM model inside the results folder.")
        st.stop()

    checkpoint = torch.load(model_path, map_location="cpu")

    word2idx = checkpoint["word2idx"]
    target_columns = checkpoint["target_columns"]
    max_len = checkpoint["max_len"]

    model = LSTMToxicClassifier(
        vocab_size=len(word2idx),
        embed_dim=128,
        hidden_dim=128,
        output_dim=len(target_columns),
        pad_idx=word2idx.get("<PAD>", 0),
        num_layers=1,
        dropout=0.3
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, word2idx, target_columns, max_len


model, word2idx, target_columns, max_len = load_model()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Prediction threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.1
)

st.sidebar.markdown("---")

st.sidebar.write("**Model:** LSTM")
st.sidebar.write("**Task:** Multi-label text classification")
st.sidebar.write("**Labels:** 6")
st.sidebar.write("**Dataset:** Jigsaw Toxic Comments")

st.sidebar.markdown("---")

st.sidebar.info(
    "Lower threshold increases recall. Higher threshold makes predictions stricter."
)


# Main UI

st.markdown('<div class="main-title">🛡️ Toxic Comment Detection App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">LSTM-based web application for detecting toxic language in online comments.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    example = st.selectbox(
        "Choose example:",
        [
            "",
            "You are so stupid and ugly.",
            "I completely agree with your opinion.",
            "I hate you.",
            "I will kill you.",
            "Thank you for your help!",
            "This article is very useful and well written."
        ]
    )

    comment = st.text_area(
        "Enter your comment:",
        value=example,
        placeholder="Type your comment here...",
        height=160
    )

    analyze = st.button("Analyze Comment", type="primary")

with col2:
    st.markdown("### Project Information")
    st.write("This app uses the trained Week 3 LSTM model.")
    st.write("It predicts probabilities for six toxic categories.")
    st.write("The final stage includes threshold tuning and error analysis.")


# Prediction

if analyze:
    if comment.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        cleaned = clean_text(comment)
        input_tensor = encode_text(comment, word2idx, max_len)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.sigmoid(logits).numpy()[0]

        results_df = pd.DataFrame({
            "Category": target_columns,
            "Probability": probabilities
        })

        results_df["Percentage"] = results_df["Probability"] * 100
        results_df["Status"] = results_df["Probability"].apply(
            lambda x: "Detected" if x >= threshold else "Not detected"
        )

        results_df = results_df.sort_values("Probability", ascending=False)

        highest_risk = float(results_df["Probability"].max())
        highest_label = results_df.iloc[0]["Category"]
        detected_count = (results_df["Probability"] >= threshold).sum()

        st.markdown("---")
        st.subheader("Overall Prediction")

        m1, m2, m3 = st.columns(3)

        m1.metric("Highest Risk", f"{highest_risk * 100:.2f}%")
        m2.metric("Top Category", highest_label)
        m3.metric("Detected Labels", int(detected_count))

        if highest_risk >= threshold:
            st.error(f"Potentially toxic comment. Highest detected category: **{highest_label}**")
        else:
            st.success("Comment looks mostly non-toxic.")

        st.subheader("Detailed Results")

        display_df = results_df.copy()
        display_df["Probability"] = display_df["Percentage"].apply(lambda x: f"{x:.2f}%")
        display_df = display_df[["Category", "Probability", "Status"]]

        st.dataframe(display_df, use_container_width=True)

        st.subheader("Probability Chart")

        chart_df = results_df.set_index("Category")[["Percentage"]]
        st.bar_chart(chart_df)

        st.subheader("Probability Bars")

        for _, row in results_df.iterrows():
            st.write(f"**{row['Category']}** — {row['Percentage']:.2f}%")
            st.progress(float(row["Probability"]))

        st.subheader("Processed Text")

        st.write("Original comment:")
        st.info(comment)

        st.write("Cleaned comment:")
        st.code(cleaned)


st.markdown("---")
st.caption("Project: Automated Detection of Cyberbullying and Toxic Behavior in Social Networks")
