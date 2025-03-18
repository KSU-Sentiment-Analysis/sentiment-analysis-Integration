import React, { useState } from 'react';
import { Container } from '@mui/material';
import Header from './components/Header';
import FeedbackInput from './components/FeedbackInput';
import SentimentResult from './components/SentimentResult';
import ResponseHistory from './components/ResponseHistory';
import Footer from './components/Footer';

const App = () => {
    const [feedback, setFeedback] = useState('');
    const [sentiment, setSentiment] = useState(null);
    const [response, setResponse] = useState('');
    const [history, setHistory] = useState([]);

    const handleAnalyze = () => {
        if (!feedback.trim()) return;

        // Simulating sentiment analysis for UI testing
        const fakeSentiments = ["Positive", "Negative", "Neutral"];
        const fakeResponses = [
            "Thank you for your kind words!",
            "We're sorry to hear that, we'll work on improving!",
            "We appreciate your feedback!"
        ];

        const randomIndex = Math.floor(Math.random() * fakeSentiments.length);
        
        setSentiment(fakeSentiments[randomIndex]);
        setResponse(fakeResponses[randomIndex]);

        setHistory([...history, { feedback, sentiment: fakeSentiments[randomIndex] }]);
        setFeedback('');
    };

    return (
        <Container maxWidth="md">
            <Header />
            <FeedbackInput feedback={feedback} setFeedback={setFeedback} handleAnalyze={handleAnalyze} />
            <SentimentResult sentiment={sentiment} response={response} />
            <ResponseHistory history={history} />
            <Footer />
        </Container>
    );
};

export default App;
