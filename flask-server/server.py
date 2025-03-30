from flask import Flask, request, jsonify, send_file
import pandas as pd
import os
from advanced_sentiment import analyze_reviews
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Need to bring over other stuff from app.py

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

        # Fill missing results
        default_result = {"advanced_sentiment": "neutral", "advanced_sentiment_score": 0,
                          "advanced_emotion": "{}", "sarcasm": False, "aspects": "{}"}

        results_filled = adv_results + [default_result] * (len(reviews) - len(adv_results))

        # Add results to DataFrame
        df["advanced_sentiment"] = [res["advanced_sentiment"] for res in results_filled]
        df["advanced_sentiment_score"] = [res["advanced_sentiment_score"] for res in results_filled]
        df["advanced_emotion"] = [res["advanced_emotion"] for res in results_filled]
        df["sarcasm_flag"] = [res["sarcasm"] for res in results_filled]
        df["aspect_analysis"] = [res["aspects"] for res in results_filled]

        # Save processed file
        processed_file_path = os.path.join(PROCESSED_FOLDER, file.filename)  # Keep original filename
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
    dataset_name = request.args.get("dataset")  # Get dataset name from query param
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    available_files = os.listdir(PROCESSED_FOLDER)
    print(f"Available Datasets in processed/: {available_files}")  # Debugging log
    print(f"Requested Dataset: {dataset_name}")  # Debugging log

    dataset_path = os.path.join(PROCESSED_FOLDER, dataset_name)

    if not os.path.exists(dataset_path):
        print(f"Dataset '{dataset_name}' not found!")  # Debugging log
        return jsonify({"error": f"Dataset '{dataset_name}' not found"}), 404

    try:
        df = pd.read_csv(dataset_path)
        return jsonify(df.to_dict(orient="records"))  # Convert DataFrame to JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
