# SMS Spam Detection Agent — Streamlit UI

A BiLSTM-based SMS spam classifier with a Streamlit front end. Given a message, the
agent predicts SPAM/HAM, shows a probability, and explains *why* using detected
trigger phrases (urgency, prizes, "click here", etc.).

## Project structure

```
sms_spam_agent/
├── app.py              # Streamlit UI (run this)
├── train_model.py       # Trains NB, LR, and BiLSTM; saves artifacts/
├── requirements.txt
└── artifacts/            # Created by train_model.py
    ├── bilstm_spam_model.keras
    ├── tokenizer.pkl
    ├── config.json
    └── model_comparison.csv
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download `spam.csv` from the [UCI SMS Spam Collection dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
   and place it in this project folder.

3. Train the model (this reproduces the original notebook pipeline and takes a
   few minutes on CPU):

   ```bash
   python train_model.py
   ```

   This creates the `artifacts/` folder with the trained BiLSTM model, tokenizer,
   config, and a comparison table of all three models (Naive Bayes, Logistic
   Regression, BiLSTM).

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

## Features

- **Single message check** — type or pick an example SMS, see prediction, spam
  probability, highlighted trigger words, and a plain-English explanation.
- **Adjustable threshold** — tune the spam probability cutoff from the sidebar.
- **Batch mode** — upload a CSV with a `text` column and classify every row at
  once, then download the results.
- **Model comparison tab** — bar chart + table comparing Naive Bayes, Logistic
  Regression, and BiLSTM on accuracy and spam F1 (from training).
- **Session history** — every single-message check is logged in a table you can
  scroll back through, with a button to clear it.

## Notes

- The app will show a warning and refuse to run predictions until `train_model.py`
  has been run at least once (it needs the saved model + tokenizer).
- If you already have `bilstm_spam_model.keras` / `tokenizer.pkl` from running the
  original notebook, you can skip `train_model.py` — just create an `artifacts/`
  folder, drop those two files in, and add a `config.json` with:
  `{"vocab_size": 5000, "max_len": 50}`
