import React, { useState } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Select,
  MenuItem,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import alertIcon from '../assets/icons8-high-risk-50-1.png';
import checklistIcon from '../assets/icons8-checklist-48.png';
import databaseIcon from '../assets/icons8-combo-chart-50-1.png';

const ResponseGeneration = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [emailResponses, setEmailResponses] = useState([]);
  const [generatedResponses, setGeneratedResponses] = useState(null);

  // Function to add an email manually (if needed)
  const handleAddEmail = () => {
    if (newEmail.trim()) {
      setEmailResponses([
        ...emailResponses,
        { email: newEmail, response: "Generated AI Response" }
      ]);
      setNewEmail('');
    }
  };

  // Function to call the Flask endpoint to generate AI responses
  const handleGenerateResponses = () => {
    if (!selectedDataset) {
      alert("Please select a dataset before generating responses.");
      return;
    }

    // Using query parameter 'dataset' matching the backend endpoint
    const endpoint = `http://localhost:5000/api/generate-responses?dataset=pre_deep_pipeline_output.csv`;

    fetch(endpoint)
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          alert(data.error);
        } else {
          // The API returns an object with "responses", "evaluation_scores", and "prompts_used"
          setGeneratedResponses(data);
        }
      })
      .catch((error) => {
        console.error("Error fetching responses:", error);
        alert("Failed to fetch response from the server.");
      });
  };

  return (
    <Container maxWidth="lg">
      {/* Page Title */}
      <Box sx={{ textAlign: "center", py: 3 }}>
        <Typography variant="h4" fontWeight="bold" color="#000">
          Response Generation
        </Typography>
      </Box>

      {/* Top Controls */}
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
          {/* Ensure these values match the actual processed CSV file names */}
          <MenuItem value="pre_deep_pipeline_output.csv">Dataset A</MenuItem>
        </Select>



        {/* Generate AI Response Button */}
        <Button
          variant="contained"
          color="secondary"
          onClick={handleGenerateResponses}
          sx={{
            borderRadius: "9999px",
            px: 4,
            fontWeight: "bold",
            whiteSpace: "nowrap",
          }}
        >
          Generate AI Responses
        </Button>
      </Paper>

      {/* Display Generated Responses (if available) */}
      {generatedResponses && (
        <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom>
            Generated Responses
          </Typography>
          {/* Responses Table */}
          <TableContainer sx={{ mt: 2 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell><strong>#</strong></TableCell>
                  <TableCell><strong>Response</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {generatedResponses.responses.map((resp, index) => (
                  <TableRow key={index}>
                    <TableCell>{index + 1}</TableCell>
                    <TableCell>{resp}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          {/* Evaluation Scores */}
          {generatedResponses.evaluation_scores && (
            <>
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                Evaluation Scores
              </Typography>
              <TableContainer sx={{ mt: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>#</strong></TableCell>
                      <TableCell><strong>Score</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {generatedResponses.evaluation_scores.map((score, index) => (
                      <TableRow key={index}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>{score}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}

          {/* Prompts Used */}
          {generatedResponses.prompts_used && (
            <>
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                Prompts Used
              </Typography>
              <TableContainer sx={{ mt: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>#</strong></TableCell>
                      <TableCell><strong>Prompt</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {generatedResponses.prompts_used.map((prompt, index) => (
                      <TableRow key={index}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>{prompt}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}
        </Paper>
      )}

      {/* Existing Dataset Section (for emails, if needed) */}
      <Paper elevation={3} sx={{ p: 3, mb: 3, borderRadius: 2 }}>
        <Typography
          variant="subtitle1"
          fontWeight="bold"
          sx={{
            color: "#1e40af",
            display: "flex",
            alignItems: "center",
            gap: 1,
          }}
        >
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
