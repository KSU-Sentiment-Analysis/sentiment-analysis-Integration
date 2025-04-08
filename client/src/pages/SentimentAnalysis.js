import React, { useState } from 'react';
import { Container, Paper, Typography, Box, Button, TextField, Rating, Grid } from '@mui/material';
import axios from 'axios';
import Papa from 'papaparse';
import BarChartIcon from '@mui/icons-material/BarChart';
import DownloadIcon from '@mui/icons-material/Download';

const SentimentAnalysis = () => {
    const [csvFile, setCsvFile] = useState(null);
    const [data, setData] = useState([]);
    const [processedFile, setProcessedFile] = useState(null);
    const [reviewText, setReviewText] = useState('');
    const [rating, setRating] = useState(3);
    const [singlePrediction, setSinglePrediction] = useState(null);
    const [metrics, setMetrics] = useState(null);

    const handleFileUpload = (event) => {
        setCsvFile(event.target.files[0]);
    };

    const handleSinglePredict = async () => {
        if (!reviewText.trim()) {
            alert("Please enter a review text");
            return;
        }

        try {
            const response = await axios.post("http://localhost:5000/predict_single", {
                review_text: reviewText,
                rating: rating
            });
            setSinglePrediction(response.data);
        } catch (error) {
            console.error("Error predicting sentiment:", error);
            alert("Failed to predict sentiment.");
        }
    };

    const handleFileSubmit = async () => {
        if (!csvFile) {
            alert("Please upload a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", csvFile);

        try {
            const uploadResponse = await axios.post("http://localhost:5000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" }
            });

            const filePath = uploadResponse.data.file_path;
            console.log("Uploaded file path:", filePath);

            const processPayload = {
                file_path: filePath,
                process_type: "advanced_pretrained"
            };

            const processResponse = await axios.post("http://localhost:5000/process", processPayload, {
                responseType: "blob"
            });

            const fileURL = URL.createObjectURL(processResponse.data);
            setProcessedFile(fileURL);

            // Get metrics for the processed file
            const metricsResponse = await axios.get("http://localhost:5000/get_metrics", {
                params: { file_path: filePath }
            });
            setMetrics(metricsResponse.data);

            Papa.parse(processResponse.data, {
                complete: (result) => {
                    setData(result.data.slice(0, 5));
                },
                header: true,
                skipEmptyLines: true
            });
        } catch (error) {
            console.error("Error processing file:", error);
            alert("Failed to process the file.");
        }
    };

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            {/* File Upload Section */}
            <Box sx={{ mb: 4 }}>
                <Paper elevation={3} sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Button
                        variant="contained"
                        component="label"
                        sx={{
                            backgroundColor: "#e0e7ff",
                            color: "#1e40af",
                            textTransform: "none",
                            fontWeight: "bold",
                            borderRadius: "9999px"
                        }}
                    >
                        Choose/Upload File
                        <input hidden type="file" accept=".csv" onChange={handleFileUpload} />
                    </Button>
                    <Typography variant="body1" sx={{ flex: 1 }}>
                        {csvFile ? csvFile.name : "File name"}
                    </Typography>
                    <select
                        style={{
                            padding: '8px',
                            borderRadius: '4px',
                            border: '1px solid #ccc'
                        }}
                    >
                        <option value="train">Train Custom Model</option>
                        <option value="custom">Use Existing Custom Model</option>
                        <option value="deep_custom">Full Deep Custom Pipeline</option>
                        <option value="advanced_pretrained">Run Pretrained Models</option>
                    </select>
                    <Button
                        variant="contained"
                        onClick={handleFileSubmit}
                        sx={{
                            borderRadius: "9999px",
                            textTransform: "none",
                            fontWeight: "bold"
                        }}
                    >
                        Process
                    </Button>
                </Paper>
            </Box>

            {/* Main Content Grid */}
            <Grid container spacing={3}>
                {/* Left Column - Metrics */}
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>                        
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                            <BarChartIcon sx={{ width: 24, height: 24 }} />
                            <Typography variant="h6">
                                Metrics Dashboard
                            </Typography>
                        </Box>
                        {metrics ? (
                            <Box sx={{ mt: 2 }}>
                                <Typography>Average Rating: {metrics.average_rating?.toFixed(2)}</Typography>
                                {processedFile && (
                                    <Button
                                        variant="outlined"
                                        startIcon={<DownloadIcon />}
                                        onClick={() => window.open(processedFile)}
                                        sx={{ mt: 1, mb: 2 }}
                                    >
                                        Download Processed File
                                    </Button>
                                )}
                                {metrics.sentiment_distribution && (
                                    <Box sx={{ mt: 2 }}>
                                        <Typography variant="subtitle2">Sentiment Distribution:</Typography>
                                        {Object.entries(metrics.sentiment_distribution).map(([sentiment, count]) => (
                                            <Typography key={sentiment}>{sentiment}: {count}</Typography>
                                        ))}
                                    </Box>
                                )}
                                {data.length > 0 && (
                                    <Box sx={{ mt: 3 }}>
                                        <Typography variant="subtitle2" gutterBottom>Recent Processed Data:</Typography>
                                        {data.map((item, index) => (
                                            <Box key={index} sx={{ mt: 1, p: 1, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                                                <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                                                    {item.text || item.review_text}
                                                </Typography>
                                            </Box>
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        ) : (
                            <Typography color="text.secondary">No metrics available</Typography>
                        )}
                    </Paper>
                </Grid>

                {/* Right Column - Single Review */}
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>                        
                        <Typography variant="h6" gutterBottom>
                            Single Review Predict
                        </Typography>
                        <TextField
                            fullWidth
                            multiline
                            rows={4}
                            variant="outlined"
                            label="Input text"
                            value={reviewText}
                            onChange={(e) => setReviewText(e.target.value)}
                            sx={{ mb: 2 }}
                        />
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                            <Typography>Ratings 1-5:</Typography>
                            <Rating
                                value={rating}
                                onChange={(event, newValue) => setRating(newValue)}
                            />
                        </Box>
                        <Button
                            variant="contained"
                            onClick={handleSinglePredict}
                            sx={{ mb: 2 }}
                        >
                            Analyze
                        </Button>
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="subtitle2" gutterBottom>Output</Typography>
                            {singlePrediction ? (
                                <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                                    <Typography variant="subtitle2">
                                        Sentiment: {singlePrediction.predicted_sentiment}
                                    </Typography>
                                    <Typography variant="subtitle2">
                                        Emotions: {Array.isArray(singlePrediction.predicted_emotions) ? singlePrediction.predicted_emotions.join(', ') : 'N/A'}
                                    </Typography>
                                    <Typography variant="subtitle2">
                                        Sarcasm: {singlePrediction.sarcasm_flag ? "Yes" : "No"}
                                    </Typography>
                                    <Typography variant="subtitle2">Aspect Analysis:</Typography>
                                        {singlePrediction.aspect_analysis && singlePrediction.aspect_analysis.length > 0 ? (
                                            singlePrediction.aspect_analysis.map((aspect, index) => (
                                            <Typography key={index}>{aspect.aspect}: {aspect.sentiment}</Typography>
                                            ))
                                        ) : (
                                            <Typography>N/A</Typography>
                                        )}
                                </Box>
                            ) : (
                                <Typography color="text.secondary">Prediction will appear here</Typography>
                            )}
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default SentimentAnalysis;
