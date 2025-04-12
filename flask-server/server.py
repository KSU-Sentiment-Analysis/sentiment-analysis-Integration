from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
import json
from flask_cors import CORS

# Import functions from your project
from dataset_handler import process_data
from custom_models import load_custom_model, train_and_save_model_from_data
from deep_pipeline import run_full_deep_analysis, predict_single_review
from gpt_response_generator import generate_structured_responses

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# rating mapping
def map_rating_to_sentiment(r):
    try:
        r = float(r)
    except Exception:
        return None
    if r > 3:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

# for uploading CSV files
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": os.path.abspath(file_path)})

# Processing endpoints
# 1."train"                -> Train Custom Model
# 1."custom"               -> Use Existing Custom Model
# 2."deep_custom"          -> Full Deep Custom Pipeline (with retraining)
# 2."advanced_pretrained"  -> Run Pretrained Advanced Models (inference only)
# 3. "Single review inference" -> as name suggests....
@app.route("/process", methods=["POST"])
def process_file():
    data = request.get_json()
    if not data or "file_path" not in data or "process_type" not in data:
        return jsonify({"error": "JSON must contain 'file_path' and 'process_type'"}), 400

    file_path = data["file_path"]
    process_type = data["process_type"]
    allowed_types = ["train", "custom", "deep_custom", "advanced_pretrained"]
    if process_type not in allowed_types:
        return jsonify({"error": f"Unsupported process_type. Must be one of {allowed_types}"}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found at the given file_path"}), 404

    # For deep pipeline options
    if process_type in ["deep_custom", "advanced_pretrained"]:
        retrain = True if process_type == "deep_custom" else False
        try:
            output_csv, metrics_json = run_full_deep_analysis(file_path, output_dir=PROCESSED_FOLDER, retrain=retrain)
            return jsonify({
                "message": "Deep pipeline analysis complete",
                "output_file": output_csv,
                "metrics_file": metrics_json
            })
        except Exception as e:
            return jsonify({"error": "Deep pipeline analysis failed", "details": str(e)}), 500

    # For custom model options
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load CSV file: {str(e)}"}), 500

    processed_df = process_data(df)
    if processed_df is None or processed_df.empty:
        return jsonify({"error": "No processed data available"}), 400

    if "rating" in processed_df.columns:
        processed_df["actual_sentiment"] = processed_df["rating"].apply(map_rating_to_sentiment)
    else:
        processed_df["actual_sentiment"] = None

    if process_type == "train":
        if "review_text" not in processed_df.columns or "rating" not in processed_df.columns:
            return jsonify({"error": "Dataset must have 'review_text' and 'rating' columns for training"}), 400
        reviews = processed_df["review_text"]
        sentiments = processed_df["rating"].apply(map_rating_to_sentiment)
        model = train_and_save_model_from_data(reviews, sentiments)
        processed_df["custom_sentiment_prediction"] = model.predict(reviews)
        out_name = "sentiment_analysis_results_custom_trained.csv"

    elif process_type == "custom":
        if "review_text" not in processed_df.columns:
            return jsonify({"error": "Dataset must have 'review_text' column for custom prediction"}), 400
        try:
            model = load_custom_model("naive_bayes_3class.pkl")
        except Exception as e:
            return jsonify({"error": "Failed to load custom model", "details": str(e)}), 500
        processed_df["custom_sentiment_prediction"] = model.predict(processed_df["review_text"])
        out_name = "sentiment_analysis_results_custom_loaded.csv"

    # Save processed CSV, for file download
    output_path = os.path.join(PROCESSED_FOLDER, out_name)
    processed_df.to_csv(output_path, index=False)
    return send_file(output_path, mimetype="text/csv", as_attachment=True, download_name=out_name)

# single review endpoint
@app.route("/predict_single", methods=["POST"])
def predict_single():
    data = request.get_json()
    if not data or "review_text" not in data or "rating" not in data:
        return jsonify({"error": "Missing review_text or rating"}), 400

    try:
        result = predict_single_review(data["review_text"], int(data["rating"]))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate-responses", methods=["GET"])
def generate_responses():
    dataset_name = request.args.get("dataset", "pre_deep_pipeline_output.csv")  # ✅ fallback value

    dataset_path = os.path.join(PROCESSED_FOLDER, dataset_name)
    if not os.path.exists(dataset_path):
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404

    try:
        df = pd.read_csv(dataset_path)
        responses, scores, prompts = generate_structured_responses(df, sample_size=5)

        return jsonify({
            "responses": responses,
            "evaluation_scores": scores,
            "prompts_used": prompts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
