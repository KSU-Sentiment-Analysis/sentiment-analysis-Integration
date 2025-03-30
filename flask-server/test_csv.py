import os
import json
import pandas as pd
from dataset_handler import process_data  # to handle original fuzzy mapping

# === Update this if needed ===
original_csv = "uploads/dataset_1.csv"
output_dir = "outputs"
final_output_path = os.path.join(output_dir, "deep_pipeline_output.csv")
final_metrics_path = os.path.join(output_dir, "deep_pipeline_metrics.json")

# === Load and preprocess base input ===
df = pd.read_csv(original_csv)
df = process_data(df)  # Ensures 'review_text' and 'rating' columns
if df is None or "review_text" not in df.columns or "rating" not in df.columns:
    raise ValueError("Processed CSV must contain 'review_text' and 'rating'")

base_df = df[["review_text", "rating"]].dropna()

# === Load individual outputs ===
sentiment_df = pd.read_csv(os.path.join(output_dir, "advanced_sentiment_output.csv"))[["review_text", "predicted_sentiment"]]
emotion_df = pd.read_csv(os.path.join(output_dir, "deep_emotion_output.csv"))[["review_text", "predicted_emotions"]]
aspect_df = pd.read_csv(os.path.join(output_dir, "aspect_sentiment_output.csv"))[["review_text", "aspect_analysis"]]
sarcasm_df = pd.read_csv(os.path.join(output_dir, "sarcasm_detection_output.csv"))[["review_text", "sarcasm_flag"]]

# === Merge everything ===
merged = base_df.merge(sentiment_df, on="review_text", how="left")
merged = merged.merge(emotion_df, on="review_text", how="left")
merged = merged.merge(aspect_df, on="review_text", how="left")
merged = merged.merge(sarcasm_df, on="review_text", how="left")

# === Save final combined output ===
merged.to_csv(final_output_path, index=False)
print(f"✅ Merged final output: {final_output_path}")

# === Reconstruct final metrics dict ===
def safe_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}

metrics_combined = {
    "sentiment_analysis": safe_load(os.path.join(output_dir, "advanced_sentiment_metrics.json")),
    "emotion_detection": safe_load(os.path.join(output_dir, "deep_emotion_metrics.json")),
    "aspect_sentiment_analysis": safe_load(os.path.join(output_dir, "aspect_sentiment_metrics.json")),
    "sarcasm_detection": safe_load(os.path.join(output_dir, "sarcasm_metrics.json"))
}

with open(final_metrics_path, "w") as f:
    json.dump(metrics_combined, f, indent=2)

print(f"📊 Final metrics saved: {final_metrics_path}")
