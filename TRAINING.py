from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch
from datasets import Dataset
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    try:
        train_df = pd.read_csv('email_data/train.csv', escapechar='\\')
        test_df = pd.read_csv('email_data/test.csv', escapechar='\\')
        
        for df in [train_df, test_df]:
            df.dropna(subset=['clean_text', 'label'], inplace=True)
            df['label'] = df['label'].astype(int)
        
        logger.info(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def tokenize_data(tokenizer, train_df, test_df):
    def tokenize(batch):
        return tokenizer(
            batch['clean_text'],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
    return train_dataset, test_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    try:
        train_df, test_df = load_data()
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        train_dataset, test_dataset = tokenize_data(tokenizer, train_df, test_df)
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        )
        
        training_args = TrainingArguments(
        output_dir='./email_model',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        disable_tqdm=False
    )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        logger.info("Starting training...")
        trainer.train()  # ✅ Do not resume unless a checkpoint exists
        
        logger.info("Running evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        trainer.save_model("email_model/final_model")
        tokenizer.save_pretrained("email_model/final_model")
        logger.info("✅ Training complete! Model saved.")
        
        return trainer
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_model()
