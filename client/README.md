# Sentiment Analysis Project

## Overview
This project consists of a **React frontend** and a **Flask backend** to analyze sentiment in textual data. It integrates **Power BI** for visualization and provides a web-based UI for uploading, processing, and retrieving datasets.

---

# 🔹 Frontend (React)

## **Setup Instructions**
### **1️⃣ Install Dependencies**
Run the following command inside the frontend folder:
```sh
npm install
```

### **2️⃣ Start the React App**
```sh
npm start
```
The app runs on **http://localhost:3000**.

## **Features**
✅ Upload CSV files containing text data
✅ Process sentiment analysis via Flask API
✅ Store and select previous datasets
✅ Embed Power BI dashboard for visualization

## **Tech Stack**
- **React.js**
- **Material UI** (for UI components)
- **Axios** (for API calls)
- **PapaParse** (for CSV parsing)
- **Power BI SDK** (for report embedding)

## **Power BI Integration**
1. **Publish your Power BI report** to Power BI Service.
2. **Get the Embed URL** from Power BI.
3. **Update `SentimentAnalysis.js`** with the correct **Embed URL, Report ID, and Access Token**.

```javascript
const embedConfig = {
    type: "report",
    id: "YOUR_REPORT_ID",
    embedUrl: "YOUR_POWERBI_EMBED_URL",
    accessToken: "YOUR_ACCESS_TOKEN",
    tokenType: models.TokenType.Embed,
};
```
4. **Restart React App** (`npm start`).

---

# 🔹 Backend (Flask)

## **Setup Instructions**
### **1️⃣ Install Dependencies**
Inside the backend folder, run:
```sh
pip install -r requirements.txt
```

### **2️⃣ Start the Flask Server**
```sh
python server.py
```
The backend runs on **http://localhost:5000**.

## **API Endpoints**
### **Upload & Process CSV**
- **Endpoint:** `POST /upload`
- **Description:** Uploads a CSV file and performs sentiment analysis.
- **Request:** FormData (`file` key with CSV file)
- **Response:** Processed CSV file as download.

### **Retrieve Processed Data**
- **Endpoint:** `GET /api/sentiment?dataset=processed_<filename>.csv`
- **Description:** Returns sentiment analysis results for a selected dataset.
- **Response:** JSON containing sentiment scores.

## **Tech Stack**
- **Flask** (Python backend framework)
- **Pandas** (Data handling)
- **Transformers (Hugging Face)** (Sentiment analysis models)
- **Flask-CORS** (Cross-origin requests)

---

# **🚀 Next Steps**
✅ Ensure the frontend & backend are correctly connected.
✅ Configure **Power BI authentication** to embed the dashboard.
✅ Deploy the application (e.g., **AWS, Azure, or Heroku**).

🎯 **For any issues, check logs in Flask (`server.py`) & React (`F12 Console`).**

🚀 Happy Coding! 🔥

