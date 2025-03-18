import React from 'react';
import { Paper, Typography } from '@mui/material';

const SentimentResult = ({ sentiment, response }) => {
    if (!sentiment) return null;

    return (
        <Paper elevation={3} sx={{ mt: 3, p: 2 }}>
            <Typography variant="h6">Sentiment: {sentiment}</Typography>
            <Typography variant="body1">Suggested Response: {response}</Typography>
        </Paper>
    );
};

export default SentimentResult;
