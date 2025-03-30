import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from dataset_handler import process_data
from pre_trained_models import load_fast_sentiment_model, load_accurate_sentiment_model
from custom_models import load_custom_model, train_and_save_model_from_data
from advanced_sentiment import analyze_reviews
from deep_pipeline import run_full_deep_analysis, predict_single_review

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def map_rating_to_sentiment(r):
    if r > 3:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"


@app.route("/")
def index():
    upload_files = os.listdir(app.config["UPLOAD_FOLDER"])
    output_files = os.listdir(app.config["OUTPUT_FOLDER"])

    upload_files_html = "<ul>"
    for filename in upload_files:
        full_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        abs_path = os.path.abspath(full_path).replace("\\", "\\\\")
        upload_files_html += f"""
            <li>
              {filename}
              <button type='button' data-filepath="{abs_path}" onclick="selectFile(this)">Use</button>
            </li>
        """
    upload_files_html += "</ul>"

    output_files_html = "<ul>"
    for filename in output_files:
        output_files_html += f'<li><a href="/download/{filename}" target="_blank">{filename}</a></li>'
    output_files_html += "</ul>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis API Test</title>
    </head>
    <body>
        <h1>Sentiment Analysis API Test</h1>

        <h2>Step 1: Upload CSV File</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" id="fileInput" accept=".csv">
            <button type="button" onclick="uploadFile()">Upload</button>
        </form>
        <div id="uploadResult" style="margin-top: 10px; color: blue;"></div>

        <h2>Step 2: Process Uploaded CSV</h2>
        <form id="processForm">
            <label for="filePath">File Path:</label>
            <input type="text" id="filePath" name="file_path" readonly style="width: 400px;">
            <br><br>
            <label for="processType">Process Type:</label>
            <select id="processType" name="process_type">
                <option value="pretrained">Pretrained</option>
                <option value="train">Train Custom Model</option>
                <option value="custom">Use Existing Custom Model</option>
                <option value="advanced">Advanced Analysis</option>
                <option value="deep_custom">Full Deep Custom Pipeline</option>
                <option value="advanced_pretrained">Run Pretrained Advanced Models</option>

            </select>
            <br><br>
            <button type="button" onclick="processFile()">Process</button>
        </form>
        <div id="processResult" style="margin-top: 10px; color: green;"></div>

        <h2>Step 3: Test a Single Review</h2>
        <textarea id="singleReview" rows="4" cols="80" placeholder="Enter your review text here"></textarea><br>
        <label for="singleRating">Rating (1-5):</label>
        <input type="number" id="singleRating" min="1" max="5">
        <br><br>
        <button type="button" onclick="predictSingle()">Analyze</button>
        <pre id="singleResult" style="background:#f4f4f4;padding:10px;"></pre>

        <h2>Existing Uploaded Files</h2>
        {upload_files_html}

        <h2>Existing Output Files (Click to Download)</h2>
        {output_files_html}

        <script>
        function selectFile(el) {{
            const path = el.getAttribute("data-filepath");
            console.log("Selected file path:", path);
            document.getElementById("filePath").value = path;
        }}

        function uploadFile() {{
            const fileInput = document.getElementById("fileInput");
            if (!fileInput.files.length) {{
                alert("Please select a file to upload.");
                return;
            }}
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            fetch("/upload", {{
                method: "POST",
                body: formData
            }})
            .then(response => response.json())
            .then(data => {{
                document.getElementById("uploadResult").innerText = JSON.stringify(data, null, 2);
                if (data.file_path) {{
                    document.getElementById("filePath").value = data.file_path;
                }}
            }})
            .catch(error => {{
                document.getElementById("uploadResult").innerText = "Error: " + error;
            }});
        }}

        function processFile() {{
            const filePath = document.getElementById("filePath").value;
            const processType = document.getElementById("processType").value;
            if (!filePath) {{
                alert("Please upload a file first or select one from the list.");
                return;
            }}

            fetch("/process", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify({{file_path: filePath, process_type: processType}})
            }})
            .then(response => response.json())
            .then(data => {{
                document.getElementById("processResult").innerText = JSON.stringify(data, null, 2);
            }})
            .catch(error => {{
                document.getElementById("processResult").innerText = "Error: " + error;
            }});
        }}

        function predictSingle() {{
            const reviewText = document.getElementById("singleReview").value;
            const rating = parseInt(document.getElementById("singleRating").value);
            if (!reviewText || !rating) {{
                alert("Enter review text and rating (1-5)");
                return;
            }}

            fetch("/predict", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify({{
                    review_text: reviewText,
                    rating: rating
                }})
            }})
            .then(res => res.json())
            .then(data => {{
                document.getElementById("singleResult").innerText = JSON.stringify(data, null, 2);
            }})
            .catch(err => {{
                document.getElementById("singleResult").innerText = "Error: " + err;
            }});
        }}
        </script>
    </body>
    </html>
    """
    return html


@app.route("/upload", methods=["POST"])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)
    return jsonify({"message": "File successfully uploaded", "file_path": os.path.abspath(saved_path)})


@app.route("/process", methods=["POST"])
def process_dataset():
    data = request.get_json()
    if not data or "file_path" not in data or "process_type" not in data:
        return jsonify({"error": "Missing 'file_path' or 'process_type' in JSON"}), 400

    file_path = data["file_path"]
    process_type = data["process_type"]

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found at the given file_path"}), 404

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

    if process_type == "pretrained":
        fast_model = load_fast_sentiment_model()
        accurate_model = load_accurate_sentiment_model()
        reviews = processed_df["review_text"].tolist()

        fast_results = fast_model(reviews, truncation=True, max_length=512)
        accurate_results = accurate_model(reviews, truncation=True, max_length=512)

        processed_df["fast_sentiment_prediction"] = [r["label"] for r in fast_results]
        processed_df["fast_sentiment_score"] = [r["score"] for r in fast_results]
        processed_df["accurate_sentiment_prediction"] = [r["label"] for r in accurate_results]
        processed_df["accurate_sentiment_score"] = [r["score"] for r in accurate_results]

        out_name = "sentiment_analysis_results_pretrained.csv"

    elif process_type == "train":
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

    elif process_type == "advanced":
        reviews = processed_df["review_text"].tolist()
        adv_results = analyze_reviews(reviews, batch_size=32)

        processed_df["advanced_sentiment"] = [res["advanced_sentiment"] for res in adv_results]
        processed_df["advanced_sentiment_score"] = [res["advanced_sentiment_score"] for res in adv_results]
        processed_df["advanced_emotion"] = [json.dumps(res["advanced_emotion"]) for res in adv_results]
        processed_df["sarcasm_flag"] = [res["sarcasm"] for res in adv_results]
        processed_df["aspect_analysis"] = [json.dumps(res["aspects"]) for res in adv_results]

        out_name = "sentiment_analysis_results_advanced.csv"

    elif process_type == "advanced_pretrained":
        try:
            # 🔁 Run deep analysis in inference mode
            output_csv, metrics_json = run_full_deep_analysis(file_path, app.config["OUTPUT_FOLDER"], retrain=False)
            return jsonify({
                "message": "Advanced pretrained inference complete",
                "output_file": output_csv,
                "metrics_file": metrics_json
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Advanced pretrained inference failed", "details": str(e)}), 500

    elif process_type == "deep_custom":
        try:
            output_csv, metrics_json = run_full_deep_analysis(file_path, app.config["OUTPUT_FOLDER"])
            return jsonify({
                "message": "Full deep custom analysis complete",
                "output_file": output_csv,
                "metrics_file": metrics_json
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Full deep custom analysis failed", "details": str(e)}), 500

    else:
        return jsonify({"error": f"Unsupported process_type: {process_type}"}), 400

    output_path = os.path.join(app.config["OUTPUT_FOLDER"], out_name)
    processed_df.to_csv(output_path, index=False)
    return jsonify({"message": "Processing complete", "output_file": os.path.abspath(output_path)})


@app.route("/predict", methods=["POST"])
def predict_single():
    data = request.get_json()
    if not data or "review_text" not in data or "rating" not in data:
        return jsonify({"error": "Missing review_text or rating"}), 400

    try:
        result = predict_single_review(data["review_text"], int(data["rating"]))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    try:
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404


# run it local :) need to do the production grade server thing not sure
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

