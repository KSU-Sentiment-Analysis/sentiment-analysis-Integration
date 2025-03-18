import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import SentimentAnalysis from './pages/SentimentAnalysis';
import ResponseGeneration from './pages/ResponseGeneration';
import LoginPage from "./pages/LoginPage";
import { AppBar, Toolbar, Tabs, Tab, Box } from '@mui/material';
import logo from './assets/capgemini_logo.png'; // Adjust path based on your project structure

const NavBar = () => {
    const location = useLocation(); // Get current route

    return (
        <AppBar position="static" sx={{ background: "linear-gradient(90deg, #1976D2, #0D47A1)" }}>

            <Toolbar>
                <Link to="/">
                    {/* Logo */}
                    <Box 
                        component="img" 
                        src={logo} 
                        alt="Logo" 
                        sx={{ 
                            height: 50, 
                            mr: 2, 
                            backgroundColor: "white",  // Adds a white background
                            borderRadius: "8px",       // Rounds the edges
                            padding: "5px",            // Adds space around the logo
                            boxShadow: "0px 2px 5px rgba(0,0,0,0.2)" // Adds a subtle shadow
                        }} 
                    />
                </Link>

                <Tabs value={location.pathname} textColor="inherit" indicatorColor="primary">
                    <Tab 
                        label="Home" 
                        component={Link} 
                        to="/" 
                        value="/" 
                        sx={{ fontWeight: location.pathname === "/" ? "bold" : "normal" }}
                    />
                    <Tab 
                        label="Sentiment Analysis" 
                        component={Link} 
                        to="/sentiment" 
                        value="/sentiment" 
                        sx={{ fontWeight: location.pathname === "/sentiment" ? "bold" : "normal" }}
                    />
                    <Tab 
                        label="Response Gen" 
                        component={Link} 
                        to="/response" 
                        value="/response" 
                        sx={{ fontWeight: location.pathname === "/response" ? "bold" : "normal" }}
                    />
                </Tabs>

                
                {/* Push Login to the right */}
                <Box sx={{ flexGrow: 1 }} /> 

                <Tab 
                    label="Login" 
                    component={Link} 
                    to="/login" 
                    value="/login" 
                    sx={{ fontWeight: location.pathname === "/login" ? "bold" : "normal" }}
                />
            </Toolbar>
        </AppBar>
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
