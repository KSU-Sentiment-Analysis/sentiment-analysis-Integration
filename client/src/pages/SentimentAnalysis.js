import React, { useState, useEffect } from 'react';
import {
    Container, Paper, Typography, Box, Button, TextField, Rating, Grid
} from '@mui/material';
import axios from 'axios';
import Papa from 'papaparse';
import BarChartIcon from '@mui/icons-material/BarChart';
import DownloadIcon from '@mui/icons-material/Download';
import {
    ResponsiveContainer, PieChart, Pie, Cell, Tooltip, 
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend
} from 'recharts';


  
const STOP_WORDS = new Set([
    'this', 'that', 'they', 'you', 'it', 'them', 'i', 'we', 'he', 'she', 'my', 'your', 'our', 'their',
    'me', 'us', 'him', 'her', 'who', 'what', 'which', 'where', 'how', 'why'
  ]);
  
const COLORS = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FF9800', '#F44336'];
const TARGET_SENTIMENTS = ['very positive', 'positive', 'neutral', 'negative', 'very negative'];

const SentimentAnalysis = () => {
    const [csvFile, setCsvFile] = useState(null);
    const [data, setData] = useState({
        sentiment: [],
        emotion: [],
        aspect: [],
      });
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

        // Hardcoded Chart Data from pre_advanced_sentiment_output.csv
        useEffect(() => {
            // FETCH SENTIMENTS
            fetch('/data/pre_advanced_sentiment_output.csv')
                .then(res => res.text())
                .then(csvText => {
                    Papa.parse(csvText, {
                        header: true,
                        skipEmptyLines: true,
                        complete: (result) => {
                            const sentimentCount = {};
                            result.data.forEach(row => {
                                const rawSentiment = row.predicted_sentiment || row.sentiment || row.Sentiment || row.label;
                                const sentiment = rawSentiment?.trim()?.toLowerCase();
                                if (sentiment) {
                                    sentimentCount[sentiment] = (sentimentCount[sentiment] || 0) + 1;
                                }
                            });
                            const sentimentData = TARGET_SENTIMENTS.map(label => ({
                                name: label,
                                value: sentimentCount[label] || 0
                            }));
                            setData(prev => ({ ...prev, sentiment: sentimentData }));
                        }
                    });
                });
        
            // FETCH EMOTIONS
            fetch('/data/pre_deep_emotion_output.csv')
                .then(res => res.text())
                .then(csvText => {
                    Papa.parse(csvText, {
                        header: true,
                        skipEmptyLines: true,
                        complete: (result) => {
                            const emotionCount = {};
                            result.data.forEach(row => {
                                const rawEmotions = row.predicted_emotions || '';
                                const emotions = rawEmotions.replace(/[\[\]']/g, '').split(',').map(e => e.trim()).filter(Boolean);
                                emotions.forEach(emotion => {
                                    emotionCount[emotion] = (emotionCount[emotion] || 0) + 1;
                                });
                            });
                            const emotionData = Object.entries(emotionCount).map(([name, value]) => ({ name, value }));
                            setData(prev => ({ ...prev, emotion: emotionData }));
                        }
                    });
                });

                // FETCH ASPECT SENTIMENT
                fetch('/data/pre_aspect_sentiment_output.csv')
                .then(res => res.text())
                .then(csvText => {
                Papa.parse(csvText, {
                    header: true,
                    skipEmptyLines: true,
                    complete: (result) => {
                    const aspectSentimentMap = {};

                    result.data.forEach(row => {
                        try {
                        const parsed = JSON.parse(row.aspect_analysis);
                        parsed.forEach(({ aspect, sentiment }) => {
                            const aspectKey = aspect.trim().toLowerCase();
                            if (STOP_WORDS.has(aspectKey)) return; // skip non-informative aspects

                            const sentimentKey = sentiment.trim().toLowerCase();

                            if (!aspectSentimentMap[aspectKey]) {
                            aspectSentimentMap[aspectKey] = {};
                            }
                            aspectSentimentMap[aspectKey][sentimentKey] =
                            (aspectSentimentMap[aspectKey][sentimentKey] || 0) + 1;
                        });
                        } catch (e) {
                        console.warn("Could not parse aspect_analysis:", row.aspect_analysis);
                        }
                    });

                    // Transform map to chart-friendly array, sorted by total sentiment count
                    const aspectData = Object.entries(aspectSentimentMap)
                    .map(([aspect, sentiments]) => ({
                    aspect,
                    ...sentiments,
                    total: Object.values(sentiments).reduce((sum, count) => sum + count, 0),
                    }))
                    .sort((a, b) => b.total - a.total)
                    .slice(0, 10) // change to 5 for Top 5 if preferred
                    .map(({ total, ...rest }) => rest); // remove 'total' before setting state

                    setData(prev => ({ ...prev, aspect: aspectData }));


                    setData(prev => ({ ...prev, aspect: aspectData }));
                    }
                });
                });

        }, []);
        
    

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            {/* Upload Section */}
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

            {/* Main Grid */}
            <Grid container spacing={3}>
                {/* Metrics Panel */}
                <Grid item xs={12} md={6}>
                    <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                            <BarChartIcon sx={{ width: 24, height: 24 }} />
                            <Typography variant="h6">
                                Sentiment Pie Chart
                            </Typography>
                        </Box>
                        {data.sentiment.length > 0 && (
                            <Box sx={{ mt: 2 }}>
                                <Typography variant="subtitle2" gutterBottom>Sentiment Distribution:</Typography>
                                <ResponsiveContainer width="100%" height={300}>
                                    <PieChart>
                                        <Pie
                                            data={data.sentiment}
                                            dataKey="value"
                                            nameKey="name"
                                            cx="50%"
                                            cy="50%"
                                            outerRadius={100}
                                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                        >
                                            {data.sentiment.map((_, index) => (
                                                <Cell key={`sentiment-${index}`} fill={COLORS[index % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                    </PieChart>
                                </ResponsiveContainer>
                            </Box>
                        )}

                        {data.emotion.length > 0 && (
                            <Box sx={{ mt: 4 }}>
                                <Typography variant="subtitle2" gutterBottom>Emotion Distribution:</Typography>
                                <ResponsiveContainer width="100%" height={300}>
                                    <PieChart>
                                        <Pie
                                            data={data.emotion}
                                            dataKey="value"
                                            nameKey="name"
                                            cx="50%"
                                            cy="50%"
                                            outerRadius={100}
                                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                        >
                                            {data.emotion.map((_, index) => (
                                                <Cell key={`emotion-${index}`} fill={COLORS[(index + 2) % COLORS.length]} />
                                            ))}
                                        </Pie>
                                        <Tooltip />
                                    </PieChart>
                                </ResponsiveContainer>
                            </Box>
                        )}

                        {data.aspect.length > 0 && (
                        <Box sx={{ mt: 4 }}>
                            <Typography variant="subtitle2" gutterBottom>Aspect-Based Sentiment Distribution:</Typography>
                            <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={data.aspect}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="aspect" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="very positive" stackId="a" fill="#4CAF50" />
                                <Bar dataKey="positive" stackId="a" fill="#8BC34A" />
                                <Bar dataKey="neutral" stackId="a" fill="#FFEB3B" />
                                <Bar dataKey="negative" stackId="a" fill="#FF9800" />
                                <Bar dataKey="very negative" stackId="a" fill="#F44336" />
                            </BarChart>
                            </ResponsiveContainer>
                        </Box>
                        )}


                    </Paper>
                </Grid>

                {/* Single Review Panel */}
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
                                        Emotions: {Array.isArray(singlePrediction.predicted_emotions)
                                            ? singlePrediction.predicted_emotions.join(', ')
                                            : 'N/A'}
                                    </Typography>
                                    <Typography variant="subtitle2">
                                        Sarcasm: {singlePrediction.sarcasm_flag ? "Yes" : "No"}
                                    </Typography>
                                    <Typography variant="subtitle2">Aspect Analysis:</Typography>
                                    {singlePrediction.aspect_analysis?.length > 0 ? (
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
