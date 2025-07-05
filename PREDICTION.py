# 📧 Email Spam Classifier
from transformers import pipeline, DistilBertTokenizerFast
import torch
import re
import sys

# ✅ Whitelist patterns (case-insensitive)
WHITELIST_PHRASES = {
    r'monthly statement',
    r'scheduled maintenance',
    r'transaction (completed|processed)',
    r'account summary',
    r'authorized payment'
}

# 🚨 High-confidence spam indicators
SPAM_TRIGGERS = {
    r'click (here|below)',
    r'verify (your|account)',
    r'limited time offer',
    r'account (suspended|locked)',
    r'http://[^\s]+'
}

def is_whitelisted(text):
    """✅ Check for legitimate banking patterns"""
    text = text.lower()
    return any(re.search(pattern, text) for pattern in WHITELIST_PHRASES)

def is_confirmed_spam(text):
    """🚨 Check for high-confidence spam triggers"""
    text = text.lower()
    return any(re.search(pattern, text) for pattern in SPAM_TRIGGERS)

def main():
    # 🚀 Load classification model
    try:
        print("Loading model...")
        classifier = pipeline(
            "text-classification",
            model="email_model/final_model",
            tokenizer="email_model/final_model",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return

    # 📥 Get email input
    print("Paste your email text (press Enter then Ctrl+Z then Enter on Windows):")
    email = sys.stdin.read()

    if not email.strip():
        print("❌ Empty input received")
        return

    # 🧹 Clean the email content
    def clean_email(text):
        text = re.sub(r'(From:|Subject:).*?\n', '', text, flags=re.IGNORECASE)
        return ' '.join(text.split())

    try:
        clean_text = clean_email(email)[:2000]
        result = classifier(clean_text, truncation=True, max_length=256)[0]

        # 🧠 Apply rule-based filtering
        if is_whitelisted(clean_text):
            final_prediction = "NOT SPAM (Whitelisted)"
        elif is_confirmed_spam(clean_text):
            final_prediction = "SPAM (Confirmed)"
        else:
            # 🔍 Use model prediction with stricter threshold
            is_spam = result['label'] == 'LABEL_1' and result['score'] > 0.7
            final_prediction = "SPAM" if is_spam else "NOT SPAM"

        # 📊 Display results
        print("\n🔍 Prediction Results:")
        print(f"Final Verdict     : {final_prediction}")
        print(f"Model Confidence  : {result['score']:.2%}")
        print(f"Raw Model Output  : {result}")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()
