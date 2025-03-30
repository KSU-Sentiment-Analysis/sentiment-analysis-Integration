# advanced_emotion.py

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

from transformers import (
    AutoTokenizer,
    pipeline,
    BertTokenizer,
    BertForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW

from dataset_handler import process_data

# Config stuff here if useful for shortning trainign time and token size
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LENGTH = 256
LR = 2e-5

EMOTION_LABELS = ["joy", "anger", "sadness", "disgust", "surprise", "fear", "neutral", "stress"]
MODEL_SAVE_PATH = "models/emotion_classifier.pt"

OUTPUT_CSV = "outputs/deep_emotion_output.csv"
METRICS_JSON = "outputs/deep_emotion_metrics.json"

#cpu/gpu dont ever use cpu or you will sitting here for ages
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#seed stuff
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Auto-labeling emotions using pre-trained pipeline might need to just rethink this whole pipline idea here

def auto_label_emotions(texts, batch_size=16):
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    emotion_pipeline = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    )

    results = []
    print("🔍 Auto-labeling emotions...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        predictions = emotion_pipeline(batch)
        for preds in predictions:
            labels = [p["label"].lower() for p in preds if p["score"] > 0.5]
            results.append(labels)
    return results

# getting ready a multi-hot dataframe
def prepare_emotion_dataframe(df, auto_labels):
    df = df.copy()
    df["emotions"] = auto_labels

    for emotion in EMOTION_LABELS:
        df[emotion] = df["emotions"].apply(lambda x: int(emotion in x))
    return df


# 70/30 split with dataset
def split_dataset(df):
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Dataset classes
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Building model set
def build_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return model.to(device)

# now train
def train_model(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=NUM_EPOCHS * len(train_loader))

    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Saved model to: {MODEL_SAVE_PATH}")

# evaluate might look into not doing this at all but w.e
def evaluate_model(model, test_loader, test_df):
    model.eval()
    all_preds = []
    all_labels = []

    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = sigmoid(outputs.logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Decode predictions
    decoded_preds = []
    for row in all_preds:
        emotion_list = [EMOTION_LABELS[i] for i, val in enumerate(row) if val == 1]
        decoded_preds.append(emotion_list)

    test_df = test_df.copy()
    test_df["predicted_emotions"] = decoded_preds
    result_df = test_df[["review_text", "predicted_emotions"]]
    result_df.to_csv(OUTPUT_CSV, index=False)

    # Metrics per emotion
    metrics = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        true = [int(label[i]) for label in all_labels]
        pred = [int(p[i]) for p in all_preds]
        metrics[emotion] = {
            "precision": precision_score(true, pred, zero_division=0),
            "recall": recall_score(true, pred, zero_division=0),
            "f1": f1_score(true, pred, zero_division=0)
        }

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved emotion output: {OUTPUT_CSV}")
    print(f"Saved metrics: {METRICS_JSON}")

    return OUTPUT_CSV, METRICS_JSON

# Flask stuff so you just need to call this method
def run_deep_emotion_analysis(csv_path, output_dir="outputs"):
    global OUTPUT_CSV, METRICS_JSON
    OUTPUT_CSV = os.path.join(output_dir, "deep_emotion_output.csv")
    METRICS_JSON = os.path.join(output_dir, "deep_emotion_metrics.json")

    df = pd.read_csv(csv_path)
    df = process_data(df)

    if df is None or "review_text" not in df.columns:
        raise ValueError("Dataset must contain 'review_text' column")

    texts = df["review_text"].tolist()
    auto_labels = auto_label_emotions(texts)

    df_labeled = prepare_emotion_dataframe(df, auto_labels)
    train_df, test_df = split_dataset(df_labeled)

    X_train = train_df["review_text"].tolist()
    y_train = train_df[EMOTION_LABELS].values.tolist()

    X_test = test_df["review_text"].tolist()
    y_test = test_df[EMOTION_LABELS].values.tolist()

    train_dataset = EmotionDataset(X_train, y_train, tokenizer)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_labels=len(EMOTION_LABELS))
    train_model(model, train_loader)
    return evaluate_model(model, test_loader, test_df)

# Inference can be used given a dataset (also need to finish the placeholder metrics path.)
def run_inference(base_df, output_dir="outputs"):

    from transformers import pipeline

    output_path = os.path.join(output_dir, "pre_deep_emotion_output.csv")
    os.makedirs(output_dir, exist_ok=True)

    if base_df is None or "review_text" not in base_df.columns:
        raise ValueError("DataFrame must have 'review_text' column")

    reviews = base_df["review_text"].tolist()

    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        tokenizer="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1
    )

    all_emotions = []
    print("Emotion Inference Running model on reviews...")
    for i in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[i:i + BATCH_SIZE]
        outputs = emotion_pipe(batch)
        for prediction in outputs:
            labels = [e["label"].lower() for e in prediction if e["score"] > 0.5]
            all_emotions.append(labels)

    result_df = pd.DataFrame({
        "review_text": reviews,
        "predicted_emotions": all_emotions
    })

    result_df.to_csv(output_path, index=False)
    print(f"Saved emotion inference results to: {output_path}")

    # Return metrics path even if not populated (for consistency)
    return result_df, METRICS_JSON

# CLI if you want to run this file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    run_deep_emotion_analysis(args.csv, output_dir=args.output_dir)
