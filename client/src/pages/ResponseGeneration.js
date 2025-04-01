import React, { useState } from 'react';
import {
    Container, Grid, Paper, Typography, Box, Button,
    Select, MenuItem, TextField, Table, TableBody, TableCell, TableContainer,
    TableHead, TableRow, Checkbox, FormControlLabel
} from '@mui/material';
import alertIcon from '../assets/icons8-high-risk-50-1.png';
import checklistIcon from '../assets/icons8-checklist-48.png';
import databaseIcon from '../assets/icons8-combo-chart-50-1.png';




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
          {/* Page Title */}
          <Box sx={{ textAlign: "center", py: 3 }}>
            <Typography variant="h4" fontWeight="bold" color="#000">
              Response Generation
            </Typography>
          </Box>
      
          {/* AI Models + Input Row */}
          <Paper
            elevation={3}
            sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                px: 4,
                py: 3,
                mb: 3,
                borderRadius: 2,
                gap: 2,
            }}
            >
            {/* AI Models Dropdown */}
            <Select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                displayEmpty
                variant="outlined"
                sx={{
                minWidth: 180,
                backgroundColor: "#e0e7ff",
                borderRadius: "9999px",
                px: 2,
                py: 1,
                fontWeight: "bold",
                color: "#1e40af",
                textTransform: "none",
                }}
            >
                <MenuItem value="" disabled>
                AI Models ⬇
                </MenuItem>
                <MenuItem value="GPT-3.5">GPT-3.5</MenuItem>
                <MenuItem value="GPT-4">GPT-4</MenuItem>
                <MenuItem value="Custom AI">Custom AI</MenuItem>
            </Select>

            {/* Select Dataset Dropdown */}
            <Select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                displayEmpty
                variant="outlined"
                sx={{
                minWidth: 180,
                backgroundColor: "#e0e7ff",
                borderRadius: "9999px",
                px: 2,
                py: 1,
                fontWeight: "bold",
                color: "#1e40af",
                textTransform: "none",
                }}
            >
                <MenuItem value="" disabled>
                Select Dataset ⬇
                </MenuItem>
                <MenuItem value="Dataset A">Dataset A</MenuItem>
                <MenuItem value="Dataset B">Dataset B</MenuItem>
                <MenuItem value="Dataset C">Dataset C</MenuItem>
            </Select>

            {/* Email Text Input */}
            <TextField
                placeholder="Insert New Email"
                variant="outlined"
                sx={{ flexGrow: 1, minWidth: 250 }}
                value={newEmail}
                onChange={(e) => setNewEmail(e.target.value)}
            />

            {/* Submit Button */}
            <Button
                variant="contained"
                color="primary"
                onClick={handleAddEmail}
                startIcon={<span style={{ fontSize: "1.2rem" }}>⏱️</span>}
                sx={{
                borderRadius: "9999px",
                px: 4,
                fontWeight: "bold",
                whiteSpace: "nowrap",
                }}
            >
                Submit
            </Button>
            </Paper>

      
          {/* Dataset Section */}
          <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 2 }}>
            <Typography variant="subtitle1" fontWeight="bold" sx={{ color: "#1e40af", display: "flex", alignItems: "center", gap: 1 }}>
            <img src={databaseIcon} alt="dataset" style={{ width: 20 }} />
              Dataset:{" "}
              <span style={{ color: "#333", fontWeight: 400 }}>
                {selectedDataset || "None Selected"}
              </span>
            </Typography>
      
            <TableContainer sx={{ mt: 2 }}>
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
      
          {/* Bottom Alert + Checklist Cards */}
          <Grid container spacing={3} sx={{ mb: 10 }}>
            <Grid item xs={6}>
                <Paper
                    sx={{
                    p: 3,
                    borderRadius: 2,
                    height: "100%",        // ✅ Ensures equal height
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "flex-start",
                    }}
                >
                    <Typography
                    variant="subtitle1"
                    fontWeight="bold"
                    color="primary"
                    sx={{ display: "flex", alignItems: "center", gap: 1 }}
                    >
                    <img src={alertIcon} alt="alert" style={{ width: 20 }} />
                    Alerts/Metric
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                    Real-time sentiment metrics will appear here.
                    </Typography>
                </Paper>
                </Grid>

                <Grid item xs={6}>
                <Paper
                    sx={{
                    p: 3,
                    borderRadius: 2,
                    height: "100%",        // ✅ Matches height of left card
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "flex-start",
                    }}
                >
                    <Typography
                    variant="subtitle1"
                    fontWeight="bold"
                    color="primary"
                    sx={{ display: "flex", alignItems: "center", gap: 1 }}
                    >
                    <img src={checklistIcon} alt="checklist" style={{ width: 20 }} />
                    Checklist (Post-Response)
                    </Typography>

                    {[1, 2, 3, 4].map((num) => (
                    <FormControlLabel
                        key={num}
                        control={<Checkbox />}
                        label={`Item #${num}`}
                        sx={{ mt: 1 }}
                    />
                    ))}
                </Paper>
            </Grid>
          </Grid>
      
          {/* Footer */}
          <Typography
            variant="caption"
            sx={{ textAlign: "center", display: "block", mt: 5, color: "gray" }}
          >
            © 2025 Heartspeak AI. All rights reserved.
          </Typography>
        </Container>
      );
      
};

export default ResponseGeneration;
