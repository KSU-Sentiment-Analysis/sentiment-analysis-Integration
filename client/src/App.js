import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { Homepage as Home } from './pages/Home';
import SentimentAnalysis from './pages/SentimentAnalysis';
import ResponseGeneration from './pages/ResponseGeneration';
import LoginPage from "./pages/LoginPage";
import Team from './pages/Team';
import { Box, Toolbar } from '@mui/material';
import logo from './assets/heartspeak-icon.png';
import themeToggleIcon from './assets/icons8-day-and-night-50-1.png';

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
                    gap: 4,
                    alignItems: 'center',
                    fontWeight: 700
                }}>
                    {[
                        { path: '/', label: 'Home' },
                        { path: '/sentiment', label: 'Sentiment Analysis' },
                        { path: '/response', label: 'Response Generation' },
                        { path: '/team', label: 'Meet The Team' }
                    ].map((item) => (
                        <Link 
                            key={item.path}
                            to={item.path} 
                            style={{ 
                                textDecoration: 'none',
                                color: location.pathname === item.path ? '#0D47A1' : '#000',
                                position: 'relative',
                                padding: '8px 0',
                                transition: 'color 0.3s ease',
                                fontWeight: 700
                            }}
                            sx={{
                                '&:hover': {
                                    color: '#1976D2'
                                },
                                '&::after': {
                                    content: '""',
                                    position: 'absolute',
                                    width: location.pathname === item.path ? '100%' : '0%',
                                    height: '2px',
                                    bottom: 0,
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    backgroundColor: '#0D47A1',
                                    transition: 'width 0.3s ease'
                                },
                                '&:hover::after': {
                                    width: '100%'
                                }
                            }}
                        >
                            {item.label}
                        </Link>
                    ))}
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
                <Route path="/team" element={<Team />} />
            </Routes>
        </Router>
    );
};

export default App;
