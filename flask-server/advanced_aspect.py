# advanced_aspect.py

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset_handler import process_data

import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW


# Config stuff here if useful for shortning trainign time and token size
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 3
MAX_LENGTH = 256
LR = 2e-5

#cpu/gpu dont ever use cpu or you will sitting here for ages
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "models/aspect_sentiment_classifier.pt"
OUTPUT_CSV = "outputs/aspect_sentiment_output.csv"
METRICS_JSON = "outputs/aspect_sentiment_metrics.json"

ASPECT_SENTIMENT_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
nlp = spacy.load("en_core_web_sm")


# Mapping rating to sentiment

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
    return None


#Extracting aspects

def extract_aspects(text):
    doc = nlp(text)
    aspects = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1]
    return list(set(aspects))


# labelling aspect-sentiment pairs (weakly) still dont work perfect haha
def create_weakly_labeled_aspect_data(df):
    aspect_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting aspects"):
        review = row["review_text"]
        sentiment = map_rating_to_sentiment(row["rating"])
        if not sentiment:
            continue

        aspects = extract_aspects(review)
        for aspect in aspects:
            aspect_rows.append({
                "aspect": aspect,
                "review_text": review,
                "sentiment": sentiment
            })

    return pd.DataFrame(aspect_rows)

# Dataset classes
class AspectDataset(Dataset):
    def __init__(self, aspects, texts, labels, tokenizer, label2id):
        self.inputs = [f"aspect: {a} [SEP] {t}" for a, t in zip(aspects, texts)]
        self.labels = [label2id[label] for label in labels]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.inputs[idx],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# label
label2id = {label: i for i, label in enumerate(ASPECT_SENTIMENT_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# Building model sets
def build_model(num_labels):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model.to(DEVICE)

# now train
def train_model(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=NUM_EPOCHS * len(train_loader)
    )

    model.train()
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            preds = torch.argmax(outputs.logits, dim=1)
            acc = (preds == labels).sum().item() / labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=acc)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Aspect model saved to: {MODEL_SAVE_PATH}")

# evaluate might look into not doing this at all but w.e
def evaluate_model(model, test_loader, test_df):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].cpu().numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    pred_labels = [id2label[p] for p in all_preds]
    true_labels = [id2label[t] for t in all_labels]

    report = classification_report(true_labels, pred_labels, output_dict=True)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": report
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved aspect sentiment metrics: {METRICS_JSON}")

    # Add predictions
    test_df = test_df.copy()
    test_df["predicted_sentiment"] = pred_labels
    return test_df

# Flask stuff so you just need to call this method
def run_aspect_analysis(csv_path, output_dir="outputs"):
    global OUTPUT_CSV, METRICS_JSON
    OUTPUT_CSV = os.path.join(output_dir, "aspect_sentiment_output.csv")
    METRICS_JSON = os.path.join(output_dir, "aspect_sentiment_metrics.json")

    df = pd.read_csv(csv_path)
    df = process_data(df)
    if df is None or "review_text" not in df.columns or "rating" not in df.columns:
        raise ValueError("Missing 'review_text' or 'rating' column")

    weak_df = create_weakly_labeled_aspect_data(df)
    train_df, test_df = train_test_split(weak_df, test_size=0.3, random_state=SEED)

    train_dataset = AspectDataset(train_df["aspect"], train_df["review_text"], train_df["sentiment"], tokenizer, label2id)
    test_dataset = AspectDataset(test_df["aspect"], test_df["review_text"], test_df["sentiment"], tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_labels=len(label2id))
    train_model(model, train_loader)
    evaluated_df = evaluate_model(model, test_loader, test_df)

    grouped = {r: [] for r in df["review_text"].unique()}

    for _, row in evaluated_df.iterrows():
        review = row["review_text"]
        grouped[review].append({
            "aspect": row["aspect"],
            "sentiment": row["predicted_sentiment"]
        })

    result_df = pd.DataFrame({
        "review_text": list(grouped.keys()),
        "aspect_analysis": [json.dumps(grouped[rev]) for rev in grouped]
    })

    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved aspect analysis: {OUTPUT_CSV}")

    return OUTPUT_CSV, METRICS_JSON

# Inference can be used given a dataset
def run_inference(base_df, output_dir="outputs"):
    """
    Run aspect-based sentiment inference using trained classifier.
    """
    output_path = os.path.join(output_dir, "pre_aspect_sentiment_output.csv")
    os.makedirs(output_dir, exist_ok=True)

    if base_df is None or "review_text" not in base_df.columns:
        raise ValueError("DataFrame must contain 'review_text' column")

    reviews = base_df["review_text"].tolist()

    # Load model
    model = build_model(num_labels=len(label2id))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_results = []

    print("Aspect Inference Extracting and predicting...")
    for review in tqdm(reviews, desc="Inferencing"):
        aspects = extract_aspects(review)
        aspect_sentiments = []

        for aspect in aspects:
            text_input = f"aspect: {aspect} [SEP] {review}"
            enc = tokenizer(text_input, return_tensors="pt", max_length=MAX_LENGTH,
                            truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
                pred_label = id2label[torch.argmax(logits).item()]
                aspect_sentiments.append({
                    "aspect": aspect,
                    "sentiment": pred_label
                })

        all_results.append(json.dumps(aspect_sentiments))

    result_df = pd.DataFrame({
        "review_text": reviews,
        "aspect_analysis": all_results
    })

    result_df.to_csv(output_path, index=False)
    print(f"Saved aspect inference results to: {output_path}")

    return result_df, METRICS_JSON


# CLI if you want to run this file
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    run_aspect_analysis(args.csv, output_dir=args.output_dir)
