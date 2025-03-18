import React, { useState } from 'react';
import {
    Container, Grid, Paper, Typography, Box, Button,
    Select, MenuItem, TextField, Table, TableBody, TableCell, TableContainer,
    TableHead, TableRow, Checkbox, FormControlLabel
} from '@mui/material';


const ResponseGeneration = () => {
    const [selectedDataset, setSelectedDataset] = useState('');
    const [newEmail, setNewEmail] = useState('');
    const [emailResponses, setEmailResponses] = useState([]);

    const handleAddEmail = () => {
        if (newEmail.trim()) {
            setEmailResponses([...emailResponses, { email: newEmail, response: "Generated AI Response" }]);
            setNewEmail('');
        }
    };

    return (
        <Container maxWidth="lg">
            {/* Centered Page Title */}
            <Box
                sx={{
                    textAlign: "center",
                    padding: "20px 0",
                }}
            >
                <Typography variant="h4" fontWeight="bold" color="primary">
                    Response Generation
                </Typography>
            </Box>

            {/* Dataset Selection and Email Input */}
            <Grid container spacing={3} sx={{ mt: 2, alignItems: 'center' }}>
                <Grid item xs={6}>
                    <Select
                        value={selectedDataset}
                        onChange={(e) => setSelectedDataset(e.target.value)}
                        displayEmpty
                        fullWidth
                    >
                        <MenuItem value="" disabled>Select Dataset</MenuItem>
                        <MenuItem value="Dataset A">Dataset A</MenuItem>
                        <MenuItem value="Dataset B">Dataset B</MenuItem>
                        <MenuItem value="Dataset C">Dataset C</MenuItem>
                    </Select>
                </Grid>

                <Grid item xs={4}>
                    <TextField
                        label="Insert New Email"
                        fullWidth
                        variant="outlined"
                        value={newEmail}
                        onChange={(e) => setNewEmail(e.target.value)}
                    />
                </Grid>

                <Grid item xs={2}>
                    <Button variant="contained" color="primary" fullWidth onClick={handleAddEmail}>
                        Submit
                    </Button>
                </Grid>
            </Grid>

            {/* Main Content Section */}
            <Grid container spacing={3} sx={{ mt: 2 }}>
                {/* Left Side - Email and Response Table */}
                <Grid item xs={8}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6">Dataset: {selectedDataset || "None Selected"}</Typography>

                        {/* Table for Emails and Responses */}
                        <TableContainer>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell><strong>User Email</strong></TableCell>
                                        <TableCell><strong>OpenAI Response</strong></TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {emailResponses.length === 0 ? (
                                        <TableRow>
                                            <TableCell colSpan={2} align="center">
                                                No emails added.
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
