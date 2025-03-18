import React, { useState } from 'react';
import { Container, Grid, Paper, Typography, Box, Button, TextField, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import axios from 'axios';
import Papa from 'papaparse';

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
        <Container maxWidth="lg">
            <Box sx={{ textAlign: "center", padding: "20px 0" }}>
                <Typography variant="h4" fontWeight="bold" color="primary">
                    Sentiment Analysis
                </Typography>
            </Box>

            {/* CSV File Upload Section */}
            <Grid container spacing={3} sx={{ mt: 2, alignItems: 'center' }}>
                <Grid item xs={8}>
                    <TextField
                        type="file"
                        fullWidth
                        inputProps={{ accept: ".csv" }}
                        onChange={handleFileUpload}
                    />
                </Grid>
                <Grid item xs={4}>
                    <Button variant="contained" color="primary" fullWidth onClick={handleFileSubmit}>
                        Upload & Process
                    </Button>
                </Grid>
            </Grid>

            {/* Display Processed Data */}
            <Grid container spacing={3} sx={{ mt: 2 }}>
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">Processed Data (Showing 5 Random Rows)</Typography>
                        {data.length > 0 ? (
                            <TableContainer>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            {Object.keys(data[0]).map((key, index) => (
                                                <TableCell key={index}><strong>{key}</strong></TableCell>
                                            ))}
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {data.map((row, rowIndex) => (
                                            <TableRow key={rowIndex}>
                                                {Object.values(row).map((value, colIndex) => (
                                                    <TableCell key={colIndex}>{value}</TableCell>
                                                ))}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        ) : (
                            <Typography variant="body2" sx={{ mt: 2, color: "gray" }}>
                                No processed data available.
                            </Typography>
                        )}
                    </Paper>
                </Grid>
            </Grid>

            {/* Download Processed CSV File */}
            {processedFile && (
                <Box sx={{ textAlign: "center", mt: 2 }}>
                    <Button variant="contained" color="secondary" href={processedFile} download="processed_results.csv">
                        Download Processed File
                    </Button>
                </Box>
            )}
        </Container>
    );
};

export default SentimentAnalysis;