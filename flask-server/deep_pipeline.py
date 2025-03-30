import os
import json
import pandas as pd
import torch

from dataset_handler import process_data
from advanced_sentiment2 import run_deep_sentiment_analysis, run_inference as sentiment_infer
from advanced_emotion import run_deep_emotion_analysis, run_inference as emotion_infer
from advanced_aspect import run_aspect_analysis, run_inference as aspect_infer
from advanced_sarcasm import run_sarcasm_analysis, run_inference as sarcasm_infer
from transformers import AutoModelForSequenceClassification as SarcasmModel

# running the whole thing in one go take an hour and some and longer depending on dataset and your gpu/cpu power
def run_full_deep_analysis(csv_path, output_dir="outputs", retrain=True):
    os.makedirs(output_dir, exist_ok=True)

    if retrain:
        print("Running Deep Sentiment Analysis (Training)...")
        sentiment_csv, sentiment_metrics = run_deep_sentiment_analysis(csv_path, output_dir)

        print("Running Deep Emotion Detection (Training)...")
        emotion_csv, emotion_metrics = run_deep_emotion_analysis(csv_path, output_dir)

        print("Running Aspect-Based Sentiment (Training)...")
        aspect_csv, aspect_metrics = run_aspect_analysis(csv_path, output_dir)

        print("Running Sarcasm Detection (Training)...")
        sarcasm_csv, sarcasm_metrics = run_sarcasm_analysis(csv_path, output_dir)

    else:
        print("Running Inference using Pretrained .pt Models...")

        # Preprocess
        raw_df = pd.read_csv(csv_path)
        base_df = process_data(raw_df)
        if base_df is None or "review_text" not in base_df.columns or "rating" not in base_df.columns:
            raise ValueError("Processed CSV must contain 'review_text' and 'rating' columns")

        base_df = base_df[["review_text", "rating"]].dropna().reset_index(drop=True)

        # Inference: Sentiment
        print("Inference: Sentiment")
        sentiment_df, sentiment_metrics = sentiment_infer(base_df, output_dir)
        sentiment_csv = os.path.join(output_dir, "pre_advanced_sentiment_output.csv")
        sentiment_df.to_csv(sentiment_csv, index=False)

        # Inference: Emotion
        print("Inference: Emotion")
        emotion_df, emotion_metrics = emotion_infer(base_df, output_dir)
        emotion_csv = os.path.join(output_dir, "pre_deep_emotion_output.csv")
        emotion_df.to_csv(emotion_csv, index=False)

        # Inference: Aspect
        print("Inference: Aspect-Based")
        aspect_df, aspect_metrics = aspect_infer(base_df, output_dir)
        aspect_csv = os.path.join(output_dir, "pre_aspect_sentiment_output.csv")
        aspect_df.to_csv(aspect_csv, index=False)

        # Inference: Sarcasm
        print("Inference: Sarcasm")
        sarcasm_df, sarcasm_metrics = sarcasm_infer(base_df, output_dir)
        sarcasm_csv = os.path.join(output_dir, "pre_sarcasm_detection_output.csv")
        sarcasm_df.to_csv(sarcasm_csv, index=False)

    # Merging all the outputs
    base_df = pd.read_csv(csv_path)
    base_df = process_data(base_df)
    base_df = base_df[["review_text", "rating"]].dropna().reset_index(drop=True)

    sentiment_df = pd.read_csv(sentiment_csv)[["review_text", "predicted_sentiment"]]
    emotion_df = pd.read_csv(emotion_csv)[["review_text", "predicted_emotions"]]
    aspect_df = pd.read_csv(aspect_csv)[["review_text", "aspect_analysis"]]
    sarcasm_df = pd.read_csv(sarcasm_csv)[["review_text", "sarcasm_flag"]]

    merged = base_df.merge(sentiment_df, on="review_text", how="left")
    merged = merged.merge(emotion_df, on="review_text", how="left")
    merged = merged.merge(aspect_df, on="review_text", how="left")
    merged = merged.merge(sarcasm_df, on="review_text", how="left")

    final_output_path = os.path.join(output_dir, "pre_deep_pipeline_output.csv")
    merged.to_csv(final_output_path, index=False)
    print(f"Final combined CSV saved: {final_output_path}")

    # Saving metrics
    metrics_combined = {}

    def try_load_metrics(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    metrics_combined["sentiment_analysis"] = try_load_metrics(sentiment_metrics)
    metrics_combined["emotion_detection"] = try_load_metrics(emotion_metrics)
    metrics_combined["aspect_sentiment_analysis"] = try_load_metrics(aspect_metrics)
    metrics_combined["sarcasm_detection"] = try_load_metrics(sarcasm_metrics)

    metrics_path = os.path.join(output_dir, "pre_deep_pipeline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_combined, f, indent=2)

    print(f"Final metrics saved: {metrics_path}")
    return final_output_path, metrics_path


# real-time single review predict :)
def predict_single_review(review_text, rating):
    from advanced_sentiment2 import build_model as build_sentiment_model, tokenizer as sentiment_tokenizer, \
        id2label as sentiment_id2label
    from advanced_emotion import EMOTION_LABELS
    from advanced_aspect import extract_aspects, build_model as build_aspect_model, tokenizer as aspect_tokenizer, \
        id2label as aspect_id2label

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sentiment
    sentiment_model = build_sentiment_model(num_labels=len(sentiment_id2label))
    sentiment_model.load_state_dict(torch.load("models/sentiment_classifier.pt", map_location=device))
    sentiment_model.to(device)
    sentiment_model.eval()
    with torch.no_grad():
        inputs = sentiment_tokenizer(review_text, return_tensors="pt", truncation=True, max_length=128,
                                     padding=True).to(device)
        logits = sentiment_model(**inputs).logits
        sentiment = sentiment_id2label[logits.argmax().item()]

    # Emotion
    from transformers import pipeline
    emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None,
                            device=0 if torch.cuda.is_available() else -1)
    emotion_output = emotion_pipe(review_text)[0]
    emotions = [e["label"].lower() for e in emotion_output if e["score"] > 0.5]

    # Sarcasm
    sarcasm_model = SarcasmModel.from_pretrained("cardiffnlp/twitter-roberta-base-irony").to(device)
    sarcasm_model.eval()
    from advanced_sarcasm import tokenizer_roberta as sarcasm_tokenizer
    encoded = sarcasm_tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(
        device)
    with torch.no_grad():
        logits = sarcasm_model(**encoded).logits
        sarcasm = bool(torch.argmax(logits).item())

    # Aspect
    aspects = extract_aspects(review_text)
    aspect_model = build_aspect_model(num_labels=len(aspect_id2label))
    aspect_model.load_state_dict(torch.load("models/aspect_sentiment_classifier.pt", map_location=device))
    aspect_model.to(device)
    aspect_model.eval()
    aspect_results = []

    for aspect in aspects:
        text_input = f"aspect: {aspect} [SEP] {review_text}"
        enc = aspect_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=128).to(
            device)
        with torch.no_grad():
            logits = aspect_model(**enc).logits
            pred_label = aspect_id2label[logits.argmax().item()]
            aspect_results.append({"aspect": aspect, "sentiment": pred_label})

    return {
        "predicted_sentiment": sentiment,
        "predicted_emotions": emotions,
        "sarcasm_flag": sarcasm,
        "aspect_analysis": aspect_results
    }
