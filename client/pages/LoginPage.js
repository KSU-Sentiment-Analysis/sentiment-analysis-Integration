import React, { useState } from "react";

const LoginPage = () => {
    const [credentials, setCredentials] = useState({ email: "", password: "" });

    const handleChange = (e) => {
        setCredentials({ ...credentials, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log("Logging in with:", credentials);
    };

    return React.createElement(
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
                    style: { width: "calc(100% - 20px)", padding: "10px", margin: "10px 0", borderRadius: "5px", border: "1px solid #ccc" },
                    onChange: handleChange
                }),
                React.createElement("input", {
                    type: "password",
                    name: "password",
                    placeholder: "Password",
                    required: true,
                    style: { width: "calc(100% - 20px)", padding: "10px", margin: "10px 0", borderRadius: "5px", border: "1px solid #ccc" },
                    onChange: handleChange
                }),
                React.createElement(
                    "button",
                    { 
                        type: "submit",
                        style: {
                            width: "100%",
                            padding: "10px",
                            backgroundColor: "#1976D2",
                            color: "white",
                            border: "none",
                            borderRadius: "5px",
                            cursor: "pointer",
                            marginTop: "10px"
                        }
                    },
                    "Login"
                )
            )
        )
    );
};

export default LoginPage;
