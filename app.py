"""
Streamlit UI for the AI Agent for SMS Spam Detection and Intelligent Message Filtering.

Run with:
    streamlit run app.py
"""

import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(
    page_title="SMS Spam Detection Agent",
    page_icon="🛡️",
    layout="centered",
)

ARTIFACTS_DIR = "artifacts"
MODEL_PATH_PT = os.path.join(ARTIFACTS_DIR, "bilstm_spam_model.pt")
MODEL_PATH_KERAS = os.path.join(ARTIFACTS_DIR, "bilstm_spam_model.keras")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
COMPARISON_PATH = os.path.join(ARTIFACTS_DIR, "model_comparison.csv")

SPAM_TRIGGER_WORDS = [
    "free", "win", "winner", "won", "prize", "cash", "urgent", "congratulations",
    "claim", "call now", "txt", "text now", "guaranteed", "offer", "click",
    "limited", "credit", "loan", "award", "selected", "collect", "reply",
    "subscribe", "voucher", "bonus", "discount", "cheap", "risk free",
]


class SimpleTokenizer:
    def __init__(self, word_index=None, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = word_index or {}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(w, 1) for w in text.split()]
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen=50):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            seq = seq[:maxlen]
        else:
            seq = seq + [0] * (maxlen - len(seq))
        padded.append(seq)
    return np.array(padded)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (ht, ct) = self.bilstm(embedded)
        out = torch.cat((ht[-2], ht[-1]), dim=1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", " httpaddr ", t)
    t = re.sub(r"\S+@\S+", " emailaddr ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def explain_prediction(cleaned_text: str, is_spam: bool):
    """Returns (explanation string, list of matched trigger words)."""
    if not is_spam:
        return "No common spam trigger words detected; message reads like normal conversation.", []
    found = sorted({w for w in SPAM_TRIGGER_WORDS if w in cleaned_text})
    if found:
        return "Flagged as spam — contains trigger phrase(s): " + ", ".join(found) + ".", found
    return (
        "Flagged as spam by the model based on overall message pattern "
        "(no single obvious trigger word, but structure/wording matches known spam patterns).",
        [],
    )


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(TOKENIZER_PATH, "rb") as f:
        tok_data = pickle.load(f)

    if isinstance(tok_data, dict):
        tokenizer = SimpleTokenizer(
            word_index=tok_data.get("word_index", {}),
            num_words=tok_data.get("num_words", 5000),
            oov_token=tok_data.get("oov_token", "<OOV>"),
        )
    else:
        tokenizer = tok_data

    if os.path.exists(MODEL_PATH_PT):
        vocab_size = config.get("vocab_size", 5000)
        model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=64, hidden_dim=32)
        model.load_state_dict(torch.load(MODEL_PATH_PT, weights_only=True))
        model.eval()
        model_type = "pytorch"
    elif os.path.exists(MODEL_PATH_KERAS):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(MODEL_PATH_KERAS)
            model_type = "keras"
        except ImportError:
            raise ImportError(
                "A .keras model artifact was found, but TensorFlow is not installed. "
                "Please run `python train_model.py` to train the PyTorch model artifact."
            )
    else:
        raise FileNotFoundError("No trained model found.")

    return model, tokenizer, config, model_type


def predict(message: str, model, tokenizer, max_len: int, threshold: float, model_type: str):
    cleaned = clean_text(message)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]), maxlen=max_len)

    if model_type == "pytorch":
        with torch.no_grad():
            tensor_seq = torch.tensor(seq, dtype=torch.long)
            prob = float(model(tensor_seq).numpy()[0][0])
    else:
        prob = float(model.predict(seq, verbose=0)[0][0])

    is_spam = prob > threshold
    explanation, triggers = explain_prediction(cleaned, is_spam)
    return {
        "message": message,
        "prediction": "SPAM" if is_spam else "HAM",
        "probability": prob,
        "explanation": explanation,
        "triggers": triggers,
    }


def highlight_triggers(message: str, triggers: list) -> str:
    highlighted = message
    for word in sorted(triggers, key=len, reverse=True):
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f"<span style='background-color:#ffcdd2;padding:1px 4px;border-radius:4px;font-weight:600'>{m.group(0)}</span>",
            highlighted,
        )
    return highlighted


def render_result(result: dict):
    is_spam = result["prediction"] == "SPAM"
    col1, col2 = st.columns([1, 2])
    with col1:
        if is_spam:
            st.error(f"🚫 {result['prediction']}")
        else:
            st.success(f"✅ {result['prediction']}")
    with col2:
        st.metric("Spam probability", f"{result['probability']*100:.2f}%")
    st.progress(min(max(result["probability"], 0.0), 1.0))

    if result["triggers"]:
        st.markdown("**Message with triggers highlighted:**")
        st.markdown(highlight_triggers(result["message"], result["triggers"]), unsafe_allow_html=True)

    st.info(f"**Why:** {result['explanation']}")


def main():
    st.title("🛡️ SMS Spam Detection Agent")
    st.caption(
        "BiLSTM-based classifier that flags spam SMS messages and explains *why*, "
        "trained on the UCI SMS Spam Collection dataset."
    )

    has_model = os.path.exists(MODEL_PATH_PT) or os.path.exists(MODEL_PATH_KERAS)
    if not (has_model and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)):
        st.warning(
            "No trained model found in `artifacts/`. Run the training script first:\n\n"
            "1. Download `spam.csv` from the UCI SMS Spam Collection dataset on Kaggle.\n"
            "2. Place it in this project folder.\n"
            "3. Run `python train_model.py`.\n"
            "4. Restart this app.",
            icon="⚠️",
        )
        st.stop()

    model, tokenizer, config, model_type = load_artifacts()
    max_len = config["max_len"]

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Spam threshold", 0.0, 1.0, 0.5, 0.05,
                               help="Probability above which a message is classified as SPAM.")
        st.divider()
        st.header("About")
        st.markdown(
            "This agent combines a **BiLSTM** deep learning model with a lightweight "
            "keyword-based explainer, so every prediction comes with a reason.\n\n"
            "**Models compared during training:** Naive Bayes (TF-IDF), "
            "Logistic Regression (TF-IDF), BiLSTM (deployed here)."
        )
        if "history" in st.session_state and st.session_state.history:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()

    if "history" not in st.session_state:
        st.session_state.history = []

    tab_single, tab_batch, tab_compare = st.tabs(["✉️ Single message", "📄 Batch (CSV)", "📊 Model comparison"])

    # --- Single message tab ---
    with tab_single:
        message = st.text_area(
            "Enter an SMS message to check:",
            placeholder="e.g. Congratulations! You have WON a $1000 gift card. Click here to claim now!!!",
            height=100,
        )
        examples = st.selectbox(
            "…or try an example",
            [
                "",
                "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now!!!",
                "Hey, are we still meeting for lunch tomorrow at 1?",
                "URGENT: Your mobile number has been awarded 500 pounds. Call 09061234567 now to collect.",
                "Can you send me the notes from today's class?",
            ],
        )
        if examples:
            message = examples

        if st.button("Analyze message", type="primary", use_container_width=True):
            if not message.strip():
                st.warning("Please enter a message first.")
            else:
                result = predict(message, model, tokenizer, max_len, threshold, model_type)
                render_result(result)
                st.session_state.history.insert(0, result)

        if st.session_state.history:
            st.divider()
            st.subheader("History")
            hist_df = pd.DataFrame(
                [{"Message": h["message"], "Prediction": h["prediction"], "Probability": f"{h['probability']*100:.2f}%"}
                 for h in st.session_state.history]
            )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # --- Batch tab ---
    with tab_batch:
        st.markdown("Upload a CSV with a column named **`text`** (one SMS message per row).")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            if "text" not in batch_df.columns:
                st.error("CSV must contain a column named 'text'.")
            else:
                if st.button("Analyze batch", type="primary"):
                    with st.spinner(f"Classifying {len(batch_df)} messages…"):
                        results = [predict(str(m), model, tokenizer, max_len, threshold, model_type) for m in batch_df["text"]]
                    out_df = pd.DataFrame([
                        {"text": r["message"], "prediction": r["prediction"],
                         "spam_probability": round(r["probability"], 4), "explanation": r["explanation"]}
                        for r in results
                    ])
                    st.dataframe(out_df, use_container_width=True, hide_index=True)
                    spam_count = (out_df["prediction"] == "SPAM").sum()
                    st.caption(f"{spam_count} of {len(out_df)} messages flagged as spam.")
                    st.download_button(
                        "Download results as CSV",
                        out_df.to_csv(index=False).encode("utf-8"),
                        file_name="spam_predictions.csv",
                        mime="text/csv",
                    )

    # --- Comparison tab ---
    with tab_compare:
        if os.path.exists(COMPARISON_PATH):
            comp_df = pd.read_csv(COMPARISON_PATH)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.bar_chart(comp_df.set_index("Model")[["Accuracy", "Spam F1"]])
        else:
            st.info("Run `train_model.py` to generate the model comparison table (Naive Bayes vs Logistic Regression vs BiLSTM).")


if __name__ == "__main__":
    main()
