import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { Homepage as Home } from './pages/Home';
import SentimentAnalysis from './pages/SentimentAnalysis';
import ResponseGeneration from './pages/ResponseGeneration';
import LoginPage from "./pages/LoginPage";
import { AppBar, Toolbar, Tabs, Tab, Box } from '@mui/material';
import logo from './assets/heartspeak-icon.png'; // Adjust path based on your project structure
import themeToggleIcon from './assets/icons8-day-and-night-50-1.png'; // Adjust path & name if needed

const NavBar = () => {
    const location = useLocation();

    return (
        <Box sx={{ boxShadow: '0 2px 4px rgba(0,0,0,0.1)', backgroundColor: '#f9f9f9', padding: '24px 160px' }}>
            <Toolbar sx={{ justifyContent: 'space-between', alignItems: 'center' }}>
                {/* Left - Logo Only */}
                <Box sx={{ display: 'flex', alignItems: 'center', height: '64px' }}>
                    <Link to="/" style={{ display: 'flex', alignItems: 'center', textDecoration: 'none' }}>
                        <Box
                            component="img"
                            src={logo}
                            alt="Logo"
                            sx={{
                                height: 130,
                                width: 'auto',
                                objectFit: 'contain',
                            }}
                        />
                    </Link>
                </Box>

                {/* Center - Navigation Tabs inside a floating box */}
                <Box sx={{
                    backgroundColor: '#fff',
                    borderRadius: '10px',
                    boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
                    px: 3,
                    py: 1,
                    display: 'flex',
                    gap: 3,
                    alignItems: 'center',
                    fontWeight: 500
                }}>
                    <Link to="/" style={{ textDecoration: 'none', color: location.pathname === '/' ? '#0D47A1' : '#000' }}>
                        Home
                    </Link>
                    <Link to="/sentiment" style={{ textDecoration: 'none', color: location.pathname === '/sentiment' ? '#0D47A1' : '#000' }}>
                        Sentiment Analysis
                    </Link>
                    <Link to="/response" style={{ textDecoration: 'none', color: location.pathname === '/response' ? '#0D47A1' : '#000' }}>
                        Response Generation
                    </Link>
                </Box>

                {/* Right - Icons and Buttons */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    {/* Placeholder for theme toggle */}
                    <Box sx={{
                        width: 36, height: 36, backgroundColor: '#fff', borderRadius: '10px',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        boxShadow: '0 1px 3px rgba(0,0,0,0.2)', cursor: 'pointer'
                    }}>
                        <Box
                            component="img"
                            src={themeToggleIcon}
                            alt="Theme Toggle"
                            sx={{
                                height: 24,
                                width: 24,
                                objectFit: 'contain',
                                cursor: 'pointer'
                            }}
                        />
                    </Box>

                    <Link to="/signup" style={{ textDecoration: 'none' }}>
                        <Box sx={{
                            backgroundColor: '#fff',
                            border: '1px solid #ccc',
                            borderRadius: '20px',
                            px: 2,
                            py: 0.5,
                            color: '#000',
                            fontWeight: 500,
                            fontSize: '14px',
                            '&:hover': { backgroundColor: '#f0f0f0' }
                        }}>
                            Sign Up
                        </Box>
                    </Link>

                    <Link to="/login" style={{ textDecoration: 'none' }}>
                        <Box sx={{
                            backgroundColor: '#0D47A1',
                            borderRadius: '20px',
                            px: 2,
                            py: 0.5,
                            color: '#fff',
                            fontWeight: 500,
                            fontSize: '14px',
                            '&:hover': { backgroundColor: '#1565c0' }
                        }}>
                            Log In
                        </Box>
                    </Link>
                </Box>
            </Toolbar>
        </Box>
    );
};




const App = () => {
    return (
        <Router>
            <NavBar />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/sentiment" element={<SentimentAnalysis />} />
                <Route path="/response" element={<ResponseGeneration />} />
                <Route path="/login" element={<LoginPage />} />
            </Routes>
        </Router>
    );
};

export default App;
