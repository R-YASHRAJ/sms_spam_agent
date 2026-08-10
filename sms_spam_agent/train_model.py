"""
Training script for the SMS Spam Detection Agent.

Trains two TF-IDF baselines (Naive Bayes, Logistic Regression) and a PyTorch BiLSTM model.
Saves all model artifacts to artifacts/ for the Streamlit UI.
"""

import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "spam.csv"
ARTIFACTS_DIR = "artifacts"
VOCAB_SIZE = 5000
MAX_LEN = 50


class SimpleTokenizer:
    def __init__(self, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        counts = {}
        for text in texts:
            for word in text.split():
                counts[word] = counts.get(word, 0) + 1
        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        self.word_index = {self.oov_token: 1}
        for word, _ in sorted_words[: self.num_words - 2]:
            self.word_index[word] = len(self.word_index) + 1
        self.index_word = {v: k for k, v in self.word_index.items()}

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


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{DATA_PATH}'. Download it from the UCI SMS Spam "
            "Collection dataset on Kaggle and place it next to this script."
        )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load & clean
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]

    df = df.dropna(subset=["text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    print("Total messages:", len(df))
    print(df["label"].value_counts())

    df["clean_text"] = df["text"].apply(clean_text).fillna("")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )
    print("Train size:", len(X_train), " Test size:", len(X_test))

    # 2. TF-IDF baselines
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    pred_nb = nb.predict(X_test_vec)
    print("\n--- Naive Bayes ---")
    print(classification_report(y_test, pred_nb, target_names=["ham", "spam"]))

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_vec, y_train)
    pred_lr = lr.predict(X_test_vec)
    print("\n--- Logistic Regression ---")
    print(classification_report(y_test, pred_lr, target_names=["ham", "spam"]))

    # 3. PyTorch BiLSTM
    tokenizer = SimpleTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.values)

    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.long),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BiLSTMClassifier(vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    model.train()
    epochs = 10
    print("\nTraining BiLSTM model...")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(X_test_seq, dtype=torch.long)
        pred_probs = model(test_inputs).numpy().flatten()

    pred_bilstm = (pred_probs > 0.5).astype(int)
    print("\n--- BiLSTM ---")
    print(classification_report(y_test, pred_bilstm, target_names=["ham", "spam"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred_bilstm))

    # 4. Comparison table
    results = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "BiLSTM"],
        "Accuracy": [
            float((pred_nb == y_test).mean()),
            float((pred_lr == y_test).mean()),
            float((pred_bilstm == y_test).mean()),
        ],
        "Spam F1": [
            float(f1_score(y_test, pred_nb)),
            float(f1_score(y_test, pred_lr)),
            float(f1_score(y_test, pred_bilstm)),
        ],
    })
    print("\n", results)
    results.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index=False)

    # 5. Save artifacts
    torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "bilstm_spam_model.pt"))
    tokenizer_data = {
        "word_index": tokenizer.word_index,
        "index_word": tokenizer.index_word,
        "num_words": tokenizer.num_words,
        "oov_token": tokenizer.oov_token,
    }
    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer_data, f)
    with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
        json.dump({"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN}, f)

    print(f"\nSaved model, tokenizer, config, and comparison table to '{ARTIFACTS_DIR}/'")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
