import React, { useState } from 'react';
import { Container, Grid, Paper, Typography, Box, Button, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import axios from 'axios';
import Papa from 'papaparse';
import yourGraphIcon from '../assets/icons8-combo-chart-50-1.png'; // Adjust path & name if needed




const SentimentAnalysis = () => {
    const [csvFile, setCsvFile] = useState(null);
    const [data, setData] = useState([]);
    const [processedFile, setProcessedFile] = useState(null);

    const handleFileUpload = (event) => {
        setCsvFile(event.target.files[0]);
    };

    const handleFileSubmit = async () => {
        if (!csvFile) {
            alert("Please upload a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", csvFile);

        try {
            const response = await axios.post("http://localhost:5000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
                responseType: "blob" // Expecting a file response
            });

            const fileURL = URL.createObjectURL(response.data);
            setProcessedFile(fileURL);

            // Parse and display processed CSV file
            Papa.parse(response.data, {
                complete: (result) => {
                    setData(result.data.slice(0, 5)); // Show 5 random rows
                },
                header: true,
                skipEmptyLines: true
            });
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Failed to process the file.");
        }
    };

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Box sx={{ textAlign: "center", mb: 4 }}>
            <Typography variant="h4" fontWeight="bold" sx={{ color: "#000" }}>
              Upload a file for Sentiment Analysis
            </Typography>
          </Box>
      
          {/* Upload Section */}
            <Paper
            elevation={3}
            sx={{
                width: "100%",
                maxWidth: "1000px",
                mx: "auto",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                px: 5,
                py: 3,
                borderRadius: 3,
                mb: 5
            }}
            >
            <Button
                variant="contained"
                component="label"
                sx={{
                backgroundColor: "#e0e7ff",
                color: "#1e40af",
                textTransform: "none",
                fontWeight: "bold",
                px: 6,
                borderRadius: "9999px"
                }}
            >
                Choose File
                <input hidden type="file" accept=".csv" onChange={handleFileUpload} />
            </Button>

            <Typography variant="body1" color="text.secondary">
                {csvFile ? csvFile.name : "No File Chosen"}
            </Typography>

            <Button
                variant="contained"
                color="primary"
                onClick={handleFileSubmit}
                startIcon={<span style={{ fontSize: "1.2rem" }}>☁️</span>}
                sx={{
                borderRadius: "9999px",
                px: 4,
                textTransform: "none",
                fontWeight: "bold"
                }}
            >
                Upload & Process
            </Button>
            </Paper>

            {/* Processed Data Box */}
            <Paper
            elevation={3}
            sx={{
                width: "100%",
                maxWidth: "1000px", // same as above
                mx: "auto",
                px: 5,
                py: 4,
                borderRadius: 3,
                display: "flex",
                flexDirection: "column",
                alignItems: "center", // centered for content
                justifyContent: "flex-start",
                textAlign: "center",
                minHeight: 200,
                mb: 5
            }}
            >
            <Typography
                variant="h6"
                fontWeight="bold"
                color="primary"
                sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mt: 0
                }}
            >
                <img
                src={yourGraphIcon}
                alt="Graph icon"
                style={{ width: 20, height: 20 }}
                />
                Processed Data (No data showing)
            </Typography>

            <Typography variant="body2" sx={{ color: "gray", mt: 0 }}>
                Visual comparisons of datasets go here.
            </Typography>
            </Paper>

      
          {/* Download Button */}
          {processedFile && (
            <Box sx={{ textAlign: "center", mt: 2 }}>
              <Button variant="contained" color="secondary" href={processedFile} download="processed_results.csv">
                Download Processed File
              </Button>
            </Box>
          )}
      
          {/* Optional Power BI Embed (keep if needed) */}
          {/* <Grid container spacing={3} sx={{ mt: 4 }}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, textAlign: "center" }}>
                <Typography variant="h6">Sentiment Analysis Report</Typography>
                <iframe
                  title="Power BI Report"
                  width="100%"
                  height="600"
                  src="..."
                  frameBorder="0"
                  allowFullScreen
                ></iframe>
              </Paper>
            </Grid>
          </Grid> */}
      
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", textAlign: "center", mt: 6 }}>
            © 2025 Heartspeak AI. All rights reserved.
          </Typography>
        </Container>
      );      
};

export default SentimentAnalysis;