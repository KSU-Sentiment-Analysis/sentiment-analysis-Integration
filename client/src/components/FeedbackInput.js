import React from 'react';
import { TextField, Button, Box } from '@mui/material';

const FeedbackInput = ({ feedback, setFeedback, handleAnalyze }) => {
    return (
        <Box sx={{ mt: 3, p: 2 }}>
            <TextField
                label="Enter your feedback"
                multiline
                rows={4}
                fullWidth
                variant="outlined"
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
            />
            <Button 
                variant="contained" 
                color="primary" 
                onClick={handleAnalyze} 
                sx={{ mt: 2, width: '100%' }}
            >
                Analyze Sentiment
            </Button>
        </Box>
    );
};

export default FeedbackInput;
