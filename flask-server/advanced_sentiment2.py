import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW

from dataset_handler import process_data

# Config stuff here if useful for shortning trainign time and token size
SEED = 42
MODEL_SAVE_PATH = "models/sentiment_classifier.pt"
OUTPUT_CSV = "outputs/advanced_sentiment_output.csv"
METRICS_JSON = "outputs/advanced_sentiment_metrics.json"
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LENGTH = 256
LR = 2e-5

#cpu/gpu dont ever use cpu or you will sitting here for ages
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#seed stuff

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#loading dataset
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df = process_data(df)

    if df is None or "review_text" not in df.columns or "rating" not in df.columns:
        raise ValueError("Dataset must contain 'review_text' and 'rating' columns")

    df = df[["review_text", "rating"]].dropna()
    df["rating"] = df["rating"].astype(int)

    def map_rating_to_sentiment(r):
        if r == 1:
            return "very negative"
        elif r == 2:
            return "negative"
        elif r == 3:
            return "neutral"
        elif r == 4:
            return "positive"
        elif r == 5:
            return "very positive"
        else:
            return None

    df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)
    df = df.dropna(subset=["sentiment"])

    return df

# 70/30 split with dataset
def split_dataset(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["sentiment"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Dataset classes
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label2id[self.labels[idx]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Tokenizer and label
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

label_list = ["very negative", "negative", "neutral", "positive", "very positive"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

#Data loaders
def get_dataloaders(train_df, test_df):
    train_dataset = SentimentDataset(
        train_df["review_text"].tolist(),
        train_df["sentiment"].tolist(),
        tokenizer,
        label2id
    )

    test_dataset = SentimentDataset(
        test_df["review_text"].tolist(),
        test_df["sentiment"].tolist(),
        tokenizer,
        label2id
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader

# Building model sets
def build_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model.to(device)

# now train
def train_model(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=NUM_EPOCHS * len(train_loader)
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).sum().item() / labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=acc)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")

# evaluate might look into not doing this at all but w.e
def evaluate_model(model, test_loader, test_df):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    pred_labels = [id2label[i] for i in all_preds]
    true_labels = [id2label[i] for i in all_labels]

    report = classification_report(true_labels, pred_labels, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    test_df = test_df.copy()
    test_df["predicted_sentiment"] = pred_labels
    test_df.to_csv(OUTPUT_CSV, index=False)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": report
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to: {OUTPUT_CSV}")
    print(f"Saved metrics to: {METRICS_JSON}")
    print(f"\nAccuracy: {accuracy:.4f} | F1: {f1:.4f}")
    if not os.path.exists(OUTPUT_CSV):
        print("Output CSV not found after writing!")
    else:
        print(f"Output CSV exists: {OUTPUT_CSV}")
    if not os.path.exists(METRICS_JSON):
        print("Metrics JSON not found after writing!")
    else:
        print(f"Metrics JSON exists: {METRICS_JSON}")

# Flask stuff so you just need to call this method
def run_deep_sentiment_analysis(csv_path, output_dir="outputs"):
    global OUTPUT_CSV, METRICS_JSON
    OUTPUT_CSV = os.path.join(output_dir, "advanced_sentiment_output.csv")
    METRICS_JSON = os.path.join(output_dir, "advanced_sentiment_metrics.json")

    df = load_and_prepare_data(csv_path)
    train_df, test_df = split_dataset(df)
    train_loader, test_loader = get_dataloaders(train_df, test_df)

    model = build_model(num_labels=len(label2id))
    train_model(model, train_loader)
    evaluate_model(model, test_loader, test_df)

    return OUTPUT_CSV, METRICS_JSON

# Inference can be used given a dataset
def run_inference(base_df, output_dir="outputs"):
    from torch.utils.data import DataLoader

    output_path = os.path.join(output_dir, "pre_advanced_sentiment_output.csv")
    os.makedirs(output_dir, exist_ok=True)

    if base_df is None or "review_text" not in base_df.columns:
        raise ValueError("Input dataframe must have 'review_text' column")

    reviews = base_df["review_text"].tolist()

    model = build_model(num_labels=len(label2id))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i:i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
            logits = model(**enc).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend([id2label[i] for i in batch_preds])

    output_df = pd.DataFrame({
        "review_text": reviews,
        "predicted_sentiment": preds
    })

    output_df.to_csv(output_path, index=False)
    print(f"Inference results saved to: {output_path}")

    return output_df, METRICS_JSON  # You can optionally compute metrics if needed


# Exporting  utilities for external use but might need to delete not sure
def get_tokenizer_and_labels():
    return tokenizer, label2id, id2label

# CLI if you want to run this file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()
    run_deep_sentiment_analysis(args.csv)
