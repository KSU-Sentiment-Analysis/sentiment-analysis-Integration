// Header.js
import React from "react";
import heartspeakIcon from "./assets/heartspeak-icon.png";
import icons8DayAndNight501 from "./assets/icons8-day-and-night-50-1.png";

import './styles/style.css';

export const Header = () => {
    return (
        <div className="header">
            <div className="bar">
                <div className="text-wrapper">Home</div>
                <div className="text-wrapper">Sentiment Analysis</div>
                <div className="text-wrapper">Response Generation</div>
            </div>

            <div className="login">
                <div className="div">Log In</div>
            </div>

            <div className="signup">
                <div className="text-wrapper-2">Sign Up</div>
            </div>

            <img
                className="heartspeak-icon"
                alt="Heartspeak icon"
                src={heartspeakIcon}
            />

            <div className="light-dark-mode">
                <img
                    className="day-and-night"
                    alt="Day and night"
                    src={icons8DayAndNight501}
                />
            </div>
        </div>
    );
};
