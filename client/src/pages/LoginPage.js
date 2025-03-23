import React, { useState } from "react";
import { Visibility, VisibilityOff } from '@mui/icons-material';

// Add this at the top of your component
const LoginPage = () => {
    // Add this new state for password visibility
    const [showPassword, setShowPassword] = useState(false);
    
    const [credentials, setCredentials] = useState({ email: "", password: "" });
    const [shake, setShake] = useState(false);
    const [isButtonPressed, setIsButtonPressed] = useState(false);

    // Add handleChange function
    const handleChange = (e) => {
        setCredentials({
            ...credentials,
            [e.target.name]: e.target.value
        });
    };

    // Add handleSubmit function
    const handleSubmit = (e) => {
        e.preventDefault();
        if (credentials.password !== "correct" || credentials.email !== "test@example.com") {
            setShake(true);
            setTimeout(() => setShake(false), 500);
            return;
        }
        console.log("Logging in with:", credentials);
    };

    // Add buttonStyle definition
    const buttonStyle = {
        width: "100%",
        padding: "10px",
        backgroundColor: "#1976D2",
        color: "white",
        border: "none",
        borderRadius: "5px",
        cursor: "pointer",
        marginTop: "10px",
        transform: isButtonPressed ? 'scale(0.95) translateY(2px)' : 'scale(1) translateY(0)',
        transition: 'all 0.2s ease',
        boxShadow: isButtonPressed 
            ? 'inset 0 3px 5px rgba(0,0,0,0.2), 0 2px 4px rgba(25, 118, 210, 0.1)' 
            : '0 6px 12px rgba(25, 118, 210, 0.3), 0 4px 6px rgba(0,0,0,0.1)',
    };

    // Update the shakeAnimation object
    const shakeAnimation = {
        animation: shake ? 'shakeEffect 0.5s' : 'none',
    };

    // Add this style element at the beginning of your return statement
    return React.createElement(
        "div",
        null,
        React.createElement(
            "style",
            null,
            `
            @keyframes shakeEffect {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
                20%, 40%, 60%, 80% { transform: translateX(10px); }
            }
            `
        ),
        React.createElement(
            "div",
            { 
                style: {
                    height: "100vh",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexDirection: "column",
                    background: "linear-gradient(90deg, #1976D2, #0D47A1)"
                }
            },
            React.createElement("h2", { style: { color: "white" } }, "HeartSpeak AI - Login"),
            React.createElement(
                "div",
                { 
                    style: {
                        background: "white",
                        padding: "20px",
                        borderRadius: "12px",
                        boxShadow: "0px 4px 10px rgba(0,0,0,0.3)",
                        textAlign: "center",
                        width: "350px"
                    } 
                },
                React.createElement("h3", { style: { color: "#1976D2" } }, "Login"),
                React.createElement(
                    "form",
                    { onSubmit: handleSubmit },
                    React.createElement("input", {
                        type: "email",
                        name: "email",
                        placeholder: "Email",
                        required: true,
                        style: { 
                            ...shakeAnimation, 
                            width: "calc(100% - 20px)", 
                            padding: "10px", 
                            margin: "10px 0", 
                            borderRadius: "5px", 
                            border: shake ? "1px solid #ff0000" : "1px solid #ccc",
                            boxShadow: shake ? "0 0 5px rgba(255, 0, 0, 0.5)" : "none"
                        },
                        onChange: handleChange
                    }),
                    React.createElement(
                        "div",
                        { 
                            style: { 
                                position: 'relative',
                                width: '100%'
                            } 
                        },
                        React.createElement("input", {
                            type: showPassword ? "text" : "password",
                            name: "password",
                            placeholder: "Password",
                            required: true,
                            style: { 
                                ...shakeAnimation, 
                                width: "calc(100% - 20px)", 
                                padding: "10px", 
                                margin: "10px 0", 
                                borderRadius: "5px", 
                                border: shake ? "1px solid #ff0000" : "1px solid #ccc",
                                boxShadow: shake ? "0 0 5px rgba(255, 0, 0, 0.5)" : "none"
                            },
                            onChange: handleChange
                        }),
                        React.createElement(
                            "button",
                            {
                                type: "button",
                                onClick: () => setShowPassword(!showPassword),
                                style: {
                                    position: 'absolute',
                                    right: '10px',
                                    top: '50%',
                                    transform: 'translateY(-50%)',
                                    border: 'none',
                                    background: 'none',
                                    cursor: 'pointer',
                                    color: '#1976D2',
                                    padding: '5px',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center'
                                }
                            },
                            React.createElement(
                                showPassword ? VisibilityOff : Visibility,
                                {
                                    style: {
                                        fontSize: '20px'
                                    }
                                }
                            )
                        )
                    ), // Remove the extra parenthesis here
                    React.createElement(
                        "button",
                        { 
                            type: "submit",
                            style: buttonStyle,
                            onMouseDown: () => setIsButtonPressed(true),
                            onMouseUp: () => setIsButtonPressed(false),
                            onMouseLeave: () => setIsButtonPressed(false)
                        },
                        "Login"
                    )
                )
            )
        )
    );
};

export default LoginPage;
