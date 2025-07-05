# email_preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

def load_enron_data(data_dir='extracted_data'):
    emails = []
    labels = []
    
    # Loop through all Enron folders
    for enron_dir in [d for d in os.listdir(data_dir) if d.startswith('enron')]:
        for label_type in ['ham', 'spam']:
            dir_path = os.path.join(data_dir, enron_dir, label_type)
            for filename in os.listdir(dir_path):
                with open(os.path.join(dir_path, filename), 'r', encoding='latin1') as f:
                    emails.append(f.read())
                    labels.append(1 if label_type == 'spam' else 0)
    
    return pd.DataFrame({'text': emails, 'label': labels})

def clean_email(text):
    # Remove headers and HTML
    text = re.sub(r'(From:|Subject:|To:|Cc:).*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace
    return ' '.join(text.split())

if __name__ == "__main__":
    # Load and clean
    df = load_enron_data()
    df['clean_text'] = df['text'].apply(clean_email)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save processed data safely
os.makedirs('email_data', exist_ok=True)
train_df.to_csv('email_data/train.csv', index=False, escapechar='\\')
test_df.to_csv('email_data/test.csv', index=False, escapechar='\\')

