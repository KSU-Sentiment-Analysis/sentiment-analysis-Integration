import React from 'react';
import { Paper, Typography, List, ListItem, ListItemText } from '@mui/material';

const ResponseHistory = ({ history }) => {
    return (
        <Paper elevation={3} sx={{ mt: 3, p: 2 }}>
            <Typography variant="h6">Feedback History</Typography>
            <List>
                {history.length === 0 ? (
                    <Typography>No history yet.</Typography>
                ) : (
                    history.map((item, index) => (
                        <ListItem key={index}>
                            <ListItemText primary={item.feedback} secondary={`Sentiment: ${item.sentiment}`} />
                        </ListItem>
                    ))
                )}
            </List>
        </Paper>
    );
};

export default ResponseHistory;
