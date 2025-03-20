import React from 'react';
import { AppBar, Toolbar, Typography } from '@mui/material';

const Header = () => {
    return (
        <AppBar position="static" color="primary">
            <Toolbar>
                <Typography variant="h5">Sentiment Analysis App</Typography>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
