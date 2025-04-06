import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import heartspeakIcon from "./assets/heartspeak-icon.png";
import icons8DayAndNight501 from "./assets/icons8-day-and-night-50-1.png";
import './styles/style.css';

export const Header = () => {
    const [isDarkMode, setIsDarkMode] = useState(localStorage.getItem('darkMode') === 'true');

    useEffect(() => {
        // Update both body class and localStorage when isDarkMode changes
        if (isDarkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        localStorage.setItem('darkMode', isDarkMode);
    }, [isDarkMode]);

    const toggleDarkMode = () => {
        setIsDarkMode(prevMode => !prevMode);
    };

    return (
        <header className={`header ${isDarkMode ? 'dark' : 'light'}`}>
            <div className="header-left">
                <div className="light-dark-mode" onClick={toggleDarkMode}>
                    <img
                        className="day-and-night"
                        alt="Toggle theme"
                        src={icons8DayAndNight501}
                    />
                </div>
                <img
                    className="heartspeak-icon"
                    alt="Heartspeak icon"
                    src={heartspeakIcon}
                />
            </div>

            <nav className="bar">
                <Link to="/" className="nav-item">Home</Link>
                <Link to="/sentiment" className="nav-item">Sentiment Analysis</Link>
                <Link to="/response" className="nav-item">Response Generation</Link>
            </nav>

            <div className="header-right">
                <button className="login-btn">Log In</button>
                <button className="signup-btn">Sign Up</button>
            </div>
        </header>
    );
};
