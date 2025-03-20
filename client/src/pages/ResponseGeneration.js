import React, { useState } from 'react';
import {
    Container, Grid, Paper, Typography, Box, Button,
    Select, MenuItem, TextField, Table, TableBody, TableCell, TableContainer,
    TableHead, TableRow, Checkbox, FormControlLabel, CircularProgress
} from '@mui/material';
import axios from 'axios';

const ResponseGeneration = () => {
    const [selectedDataset, setSelectedDataset] = useState(localStorage.getItem("uploadedDataset") || '');

    const [newEmail, setNewEmail] = useState('');
    const [emailResponses, setEmailResponses] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleGenerateResponses = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.get(`http://localhost:5000/api/generate-responses?dataset=${selectedDataset}`);

            if (response.data.responses) {
                const responsesFromApi = response.data.responses;

                // Map each response from GPT into state
                const formattedResponses = responsesFromApi.map((res, index) => ({
                    email: `Review ${index + 1}`,
                    response: res
                }));

                setEmailResponses(formattedResponses);
            }
        } catch (err) {
            console.error("Error fetching responses:", err);
            setError("Failed to generate responses.");
        }
        setLoading(false);
    };

    return (
        <Container maxWidth="lg">
            {/* Centered Page Title */}
            <Box sx={{ textAlign: "center", padding: "20px 0" }}>
                <Typography variant="h4" fontWeight="bold" color="primary">
                    Response Generation
                </Typography>
            </Box>

            {/* Dataset Selection and Response Generation Trigger */}
            <Grid container spacing={3} sx={{ mt: 2, alignItems: 'center' }}>
                <Grid item xs={8}>
                    <Select
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                        fullWidth
                    >
                        <MenuItem value="processed_results.csv">processed_results.csv</MenuItem>
                        {/* Add more options dynamically if you have multiple datasets */}
                    </Select>
                </Grid>
                <Grid item xs={4}>
                    <Button variant="contained" color="primary" fullWidth onClick={handleGenerateResponses}>
                        Generate AI Responses
                    </Button>
                </Grid>
            </Grid>

            {/* Loading indicator */}
            {loading && (
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                    <CircularProgress />
                </Box>
            )}

            {/* Error handling */}
            {error && (
                <Typography color="error" align="center" sx={{ mt: 2 }}>
                    {error}
                </Typography>
            )}

            {/* Main Content Section */}
            <Grid container spacing={3} sx={{ mt: 2 }}>
                {/* Left Side - Email and Response Table */}
                <Grid item xs={8}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">Responses for: {selectedDataset}</Typography>

                        {/* Table for Emails and Responses */}
                        <TableContainer>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell><strong>Review</strong></TableCell>
                                        <TableCell><strong>OpenAI Response</strong></TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {emailResponses.length === 0 ? (
                                        <TableRow>
                                            <TableCell colSpan={2} align="center">
                                                No responses generated yet.
                                            </TableCell>
                                        </TableRow>
                                    ) : (
                                        emailResponses.map((row, index) => (
                                            <TableRow key={index}>
                                                <TableCell>{row.email}</TableCell>
                                                <TableCell>{row.response}</TableCell>
                                            </TableRow>
                                        ))
                                    )}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Paper>
                </Grid>

                {/* Right Side - Alerts Box */}
                <Grid item xs={4}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">Alerts / Metrics Box</Typography>
                        <Typography variant="body2">
                            Important alerts and performance metrics will be displayed here.
                        </Typography>

                        {/* Custom Checklist */}
                        <Typography variant="h6" sx={{ mt: 2 }}>Checklist (Post-Response)</Typography>
                        {[1, 2, 3, 4].map((num) => (
                            <FormControlLabel
                                key={num}
                                control={<Checkbox />}
                                label={`Checklist Item ${num}`}
                            />
                        ))}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default ResponseGeneration;
