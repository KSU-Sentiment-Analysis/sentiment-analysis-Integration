# advanced_sarcasm.py

import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from dataset_handler import process_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, precision_score, recall_score
)
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    BertTokenizer, BertForSequenceClassification, get_scheduler
)
from torch.optim import AdamW

# Config stuff here if useful for shortning trainign time and token size
PRETRAINED_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: False, 1: True}

TRAINED_MODEL_PATH = "models/sarcasm_classifier.pt"
OUTPUT_CSV = "outputs/sarcasm_detection_output.csv"
METRICS_JSON = "outputs/sarcasm_metrics.json"

BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 256
SEED = 42

# tokenizer parts
tokenizer_roberta = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")

label2id = {False: 0, True: 1}
id2label = {0: False, 1: True}


# Generating pseudo labels same as aspect style might need to change this whole process
@torch.no_grad()
def predict_sarcasm(texts, batch_size=32):
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)
    model.eval()
    tokenizer = tokenizer_roberta

    flags = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting Sarcasm"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(DEVICE)
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        flags.extend([LABEL_MAP[p] for p in preds])
    return flags

def generate_pseudo_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = process_data(df)
    if df is None or "review_text" not in df.columns:
        raise ValueError("Missing 'review_text' column")

    reviews = df["review_text"].tolist()
    df["sarcasm_flag"] = predict_sarcasm(reviews)
    return df

# Dataset class
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Building model sets
def build_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        id2label={0: "not_sarcastic", 1: "sarcastic"},
        label2id={"not_sarcastic": 0, "sarcastic": 1}
    )
    return model.to(DEVICE)

# training
def train_model(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                              num_training_steps=NUM_EPOCHS * len(train_loader))

    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        total, correct = 0, 0
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item(), acc=correct/total)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print(f"Saved sarcasm model: {TRAINED_MODEL_PATH}")

# evaluate might look into not doing this at all but w.e
def evaluate_model(model, test_loader, test_df):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Sarcasm Classifier"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    pred_flags = [bool(p) for p in all_preds]
    true_flags = [bool(t) for t in all_labels]

    report = classification_report(true_flags, pred_flags, output_dict=True)
    accuracy = accuracy_score(true_flags, pred_flags)
    f1 = f1_score(true_flags, pred_flags)
    precision = precision_score(true_flags, pred_flags)
    recall = recall_score(true_flags, pred_flags)

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": report
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved sarcasm metrics: {METRICS_JSON}")

    test_df = test_df.copy()
    test_df["sarcasm_flag"] = pred_flags

    result_df = test_df[["review_text", "sarcasm_flag"]].drop_duplicates()
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved sarcasm output: {OUTPUT_CSV}")

    return OUTPUT_CSV, METRICS_JSON

# Flask stuff so you just need to call this method
def run_sarcasm_analysis(csv_path, output_dir="outputs"):
    global OUTPUT_CSV, METRICS_JSON
    OUTPUT_CSV = os.path.join(output_dir, "sarcasm_detection_output.csv")
    METRICS_JSON = os.path.join(output_dir, "sarcasm_metrics.json")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating pseudo labels...")
    df = generate_pseudo_labels(csv_path)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["sarcasm_flag"])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_dataset = SarcasmDataset(train_df["review_text"], train_df["sarcasm_flag"].map(label2id), tokenizer_bert)
    test_dataset = SarcasmDataset(test_df["review_text"], test_df["sarcasm_flag"].map(label2id), tokenizer_bert)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Fine-tuning sarcasm detector...")
    model = build_model()
    train_model(model, train_loader)

    print("Evaluating sarcasm model...")
    _, _ = evaluate_model(model, test_loader, test_df)


    df[["review_text", "sarcasm_flag"]].drop_duplicates().to_csv(OUTPUT_CSV, index=False)
    print(f"Final sarcasm output saved: {OUTPUT_CSV}")

    return OUTPUT_CSV, METRICS_JSON

# Inference can be used given a dataset
def run_inference(base_df, output_dir="outputs"):
    output_path = os.path.join(output_dir, "pre_sarcasm_detection_output.csv")
    os.makedirs(output_dir, exist_ok=True)

    if base_df is None or "review_text" not in base_df.columns:
        raise ValueError("DataFrame must contain 'review_text' column")

    reviews = base_df["review_text"].tolist()

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    predictions = []

    print("Sarcasm Inference Predicting sarcasm...")
    with torch.no_grad():
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i:i + BATCH_SIZE]
            enc = tokenizer_bert(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            flags = [bool(p) for p in preds]
            predictions.extend(flags)

    result_df = pd.DataFrame({
        "review_text": reviews,
        "sarcasm_flag": predictions
    })

    result_df.to_csv(output_path, index=False)
    print(f"Saved sarcasm inference results to: {output_path}")

    return result_df, METRICS_JSON


# CLI if you want to run this file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    run_sarcasm_analysis(args.csv, output_dir=args.output_dir)
