import React from 'react';
import { Typography, Box } from '@mui/material';

const Footer = () => {
    return (
        <Box sx={{ mt: 5, p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="textSecondary">
                © 2025 Sentiment Analysis App. All Rights Reserved.
            </Typography>
        </Box>
    );
};

export default Footer;
