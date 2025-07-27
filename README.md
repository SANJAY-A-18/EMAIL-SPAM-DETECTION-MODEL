# 📧 Email Spam Detection System (BERT + Rules)

This project is an end-to-end **email spam detection system** that combines the power of **transformer-based deep learning** with **custom rule-based filters** for enhanced accuracy.

Designed for practical use, it takes raw email text input and classifies it as **SPAM** or **NOT SPAM** based on:
- Fine-tuned **DistilBERT** model
- Whitelisting and high-confidence spam rules

---

## 🚀 Key Features

- 🔍 Preprocessing pipeline for Enron dataset
- 🤖 Spam detection using `DistilBERT`
- ⚖️ Custom rule-based filters for known spam/ham patterns
- 🧪 Evaluation with metrics: Accuracy, Precision, Recall, F1-score
- 🖥️ Command-line prediction script

---

## 🧱 Project Structure

```bash
├── PREPROCESSING.py       # Cleans Enron data and splits train/test sets
├── TRAINING.py            # Fine-tunes DistilBERT on cleaned e-mail data
├── PREDICTION.py          # Loads trained model and classifies input e-mails
├── email_model/             # Contains final fine-tuned model
└── email_data/              # CSVs for training and testing
