import openai
from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import random
import json
from openai import AzureOpenAI
from flask_cors import CORS
from advanced_sentiment import analyze_reviews
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load environment variables and configure Azure OpenAI
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Must match env var name
    api_version="2024-08-01-preview",          # Supported version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load CSV & standardize column names
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        if "review_text" not in df.columns:
            return "Missing 'Review Text' column in CSV", 400

        # Convert to string and remove NaNs
        df["review_text"] = df["review_text"].astype(str).fillna("")

        # Perform sentiment analysis
        reviews = df["review_text"].tolist()
        adv_results = analyze_reviews([r for r in reviews if r.strip() != ""], batch_size=32)

        # Fill missing results if some reviews did not get analyzed
        default_result = {"advanced_sentiment": "neutral",
                          "advanced_sentiment_score": 0,
                          "advanced_emotion": "{}",
                          "sarcasm": False,
                          "aspects": "{}"}
        results_filled = adv_results + [default_result] * (len(reviews) - len(adv_results))

        # Add results to DataFrame
        df["advanced_sentiment"] = [res["advanced_sentiment"] for res in results_filled]
        df["advanced_sentiment_score"] = [res["advanced_sentiment_score"] for res in results_filled]
        df["advanced_emotion"] = [res["advanced_emotion"] for res in results_filled]
        df["sarcasm_flag"] = [res["sarcasm"] for res in results_filled]
        df["aspect_analysis"] = [res["aspects"] for res in results_filled]

        # Save processed file
        processed_file_path = os.path.join(PROCESSED_FOLDER, file.filename)
        df.to_csv(processed_file_path, index=False)

        print(f"Processed file saved: {processed_file_path}")
        return send_file(processed_file_path, mimetype="text/csv", as_attachment=True, download_name="processed_results.csv")

    except Exception as e:
        import traceback
        print("Error processing file:", e)
        traceback.print_exc()
        return f"Internal server error: {e}", 500

@app.route("/api/sentiment", methods=["GET"])
def get_sentiment_data():
    dataset_name = request.args.get("dataset")
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    dataset_path = os.path.join(PROCESSED_FOLDER, dataset_name)
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404

    try:
        df = pd.read_csv(dataset_path)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate-responses", methods=["GET"])
def generate_responses():
    dataset_name = request.args.get("dataset")
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    dataset_path = os.path.join(PROCESSED_FOLDER, dataset_name)
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404

    try:
        df = pd.read_csv(dataset_path)

        # Randomly sample 5 reviews (or less if dataset has fewer entries)
        sampled_df = df.sample(n=min(5, len(df)))

        # Construct a structured prompt for the GPT model
        prompt = (
            "You're a customer service agent for a clothing brand. "
            "Generate empathetic, professional responses to these customer reviews:\n\n"
        )
        for idx, row in sampled_df.iterrows():
            # Parse the advanced emotion field; if it's a string, load it as JSON
            try:
                emotions = json.loads(row["advanced_emotion"]) if isinstance(row["advanced_emotion"], str) else row["advanced_emotion"]
            except Exception:
                emotions = {}
            primary_emotion = max(emotions, key=emotions.get) if emotions else "neutral"

            prompt += (
                f"Review {idx+1}:\n"
                f"Text: \"{row['review_text']}\"\n"
                f"Rating: {row.get('rating', 'N/A')}/5\n"
                f"Sentiment: {row['advanced_sentiment']}\n"
                f"Emotion: {primary_emotion.capitalize()}\n"
                f"Sarcasm Detected: {'Yes' if row['sarcasm_flag'] else 'No'}\n"
                f"Aspects: {row['aspect_analysis']}\n\n"
            )

        prompt += "Provide a thoughtful and empathetic response for each review individually:\n"

        # Call Azure OpenAI GPT API
        response = client.chat.completions.create(
            model="sentiment-openai-instance",  # Deployment name
            messages=[
                {
                    "role": "system",
                    "content": "You are an empathetic customer service representative who responds professionally to customer feedback."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )


        generated_text = response.choices[0].message.content.strip()

        # Assume that GPT separates each review response by two newline characters
        individual_responses = generated_text.split("\n\n")

        return jsonify({
            "responses": individual_responses,
            "prompt_used": prompt
        })

    except Exception as e:
        print(f"Error during GPT response generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)