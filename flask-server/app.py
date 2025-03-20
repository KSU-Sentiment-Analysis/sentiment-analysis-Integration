import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from dataset_handler import process_data
from pre_trained_models import load_fast_sentiment_model, load_accurate_sentiment_model
from custom_models import load_custom_model, train_and_save_model_from_data
from advanced_sentiment import analyze_reviews

app = Flask(__name__)

# Folder config stuff and max upload size
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

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
    """
    Basic html page for testing out of terminal
      1. CSV file upload
      2. Selection of existing CSV files (with a 'Use' button)
      3. Choosing a process type (pretrained, train, custom, advanced)
      4. Viewing and downloading output files
    """

    upload_files = os.listdir(app.config["UPLOAD_FOLDER"])
    output_files = os.listdir(app.config["OUTPUT_FOLDER"])


    upload_files_html = "<ul>"
    for filename in upload_files:
        full_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        abs_path = os.path.abspath(full_path)
        safe_path = abs_path.replace("\\", "\\\\")
        # Storing path in data-filepath atr so we dont lose it
        upload_files_html += f"""
            <li>
              {filename}
              <button type='button' data-filepath="{safe_path}" onclick="selectFile(this)">Use</button>
            </li>
        """
    upload_files_html += "</ul>"

    # making html list for the output fiels to be able to download them
    output_files_html = "<ul>"
    for filename in output_files:
        output_files_html += f'<li><a href="/download/{filename}" target="_blank">{filename}</a></li>'
    output_files_html += "</ul>"

    # HTML page for testing for now
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sentiment Analysis API Test</title>
    </head>
    <body>
        <h1>Sentiment Analysis API Test</h1>

        <!-- Step 1: Upload CSV File -->
        <h2>Step 1: Upload CSV File</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" id="fileInput" accept=".csv">
            <button type="button" onclick="uploadFile()">Upload</button>
        </form>
        <div id="uploadResult" style="margin-top: 10px; color: blue;"></div>

        <!-- Step 2: Process a CSV File -->
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
            </select>
            <br><br>
            <button type="button" onclick="processFile()">Process</button>
        </form>
        <div id="processResult" style="margin-top: 10px; color: green;"></div>

        <!-- Existing Uploaded Files -->
        <h2>Existing Uploaded Files</h2>
        {upload_files_html}

        <!-- Existing Output Files (download links) -->
        <h2>Existing Output Files (Click to Download)</h2>
        {output_files_html}

        <script>
        // Populate filePath input from data-filepath
        function selectFile(el) {{
            const path = el.getAttribute("data-filepath");
            console.log("Selected file path:", path);
            document.getElementById("filePath").value = path;
        }}

        // Upload a new CSV file
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

        // Process the selected CSV file
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
        </script>
    </body>
    </html>
    """
    return html


@app.route("/upload", methods=["POST"])
def upload_csv():
    """
    upload csv file and return file path
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    filename = secure_filename(file.filename)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(saved_path)

    # Converting here to absolute path
    absolute_path = os.path.abspath(saved_path)
    return jsonify({"message": "File successfully uploaded", "file_path": absolute_path})


@app.route("/process", methods=["POST"])
def process_dataset():
    """
    Processign the dataset using four of these options
      1. Pretrained models
      2. Train a new custom model
      3. Use an existing custom model
      4. Advanced sentiment analysis
    """
    data = request.get_json()
    if not data or "file_path" not in data or "process_type" not in data:
        return jsonify({"error": "Missing 'file_path' or 'process_type' in JSON"}), 400

    file_path = data["file_path"]
    process_type = data["process_type"]

    # Check if the file actually exists
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found at the given file_path"}), 404

    # Loads CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load CSV file: {str(e)}"}), 500

    # Here is where I Process DF using dataset_handler.py
    processed_df = process_data(df)
    if processed_df is None or processed_df.empty:
        return jsonify({"error": "No processed data available"}), 400

    if "rating" in processed_df.columns:
        processed_df["actual_sentiment"] = processed_df["rating"].apply(map_rating_to_sentiment)
    else:
        processed_df["actual_sentiment"] = None

    # naming scheme need to update !!!
    if process_type == "pretrained":
        # Option 1: Pretrained models
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
        # Option 2
        if "review_text" not in processed_df.columns or "rating" not in processed_df.columns:
            return jsonify({"error": "Dataset must have 'review_text' and 'rating' columns for training"}), 400

        reviews = processed_df["review_text"]
        sentiments = processed_df["rating"].apply(map_rating_to_sentiment)

        model = train_and_save_model_from_data(reviews, sentiments)
        custom_predictions = model.predict(reviews)
        processed_df["custom_sentiment_prediction"] = custom_predictions

        out_name = "sentiment_analysis_results_custom_trained.csv"

    elif process_type == "custom":
        # Option 3:
        if "review_text" not in processed_df.columns:
            return jsonify({"error": "Dataset must have 'review_text' column for custom prediction"}), 400

        reviews = processed_df["review_text"]
        try:
            model = load_custom_model("naive_bayes_3class.pkl")
        except Exception as e:
            return jsonify({"error": "Failed to load custom model", "details": str(e)}), 500

        custom_predictions = model.predict(reviews)
        processed_df["custom_sentiment_prediction"] = custom_predictions

        out_name = "sentiment_analysis_results_custom_loaded.csv"

    elif process_type == "advanced":
        # Option 4:
        reviews = processed_df["review_text"].tolist()
        adv_results = analyze_reviews(reviews, batch_size=32)

        processed_df["advanced_sentiment"] = [res["advanced_sentiment"] for res in adv_results]
        processed_df["advanced_sentiment_score"] = [res["advanced_sentiment_score"] for res in adv_results]
        processed_df["advanced_emotion"] = [json.dumps(res["advanced_emotion"]) for res in adv_results]
        processed_df["sarcasm_flag"] = [res["sarcasm"] for res in adv_results]
        processed_df["aspect_analysis"] = [json.dumps(res["aspects"]) for res in adv_results]

        out_name = "sentiment_analysis_results_advanced.csv"

    else:
        return jsonify({"error": "Invalid process_type. Must be 'pretrained', 'train', 'custom', or 'advanced'"}), 400

    # Here is making the path
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], out_name)
    processed_df.to_csv(output_path, index=False)

    return jsonify({"message": "Processing complete", "output_file": os.path.abspath(output_path)})


@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    """
    Downloads one of the output files
    """
    try:
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404


if __name__ == "__main__":
    # Run locally on this http://localhost:5000
    app.run(host="0.0.0.0", port=5000)
