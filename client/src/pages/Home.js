import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Typography, Button } from '@mui/material';
import { ChevronLeft, ChevronRight } from '@mui/icons-material';
import { motion } from 'framer-motion';
import { Box } from "@mui/material";

const reviews = [
    { name: "John Doe", rating: "⭐⭐⭐⭐⭐", review: "Amazing experience! The service was top-notch, highly recommend it." },
    { name: "Jane Smith", rating: "⭐⭐⭐⭐", review: "Great quality and friendly support. A little pricey, but worth it!" },
    { name: "Michael Johnson", rating: "⭐⭐⭐⭐⭐", review: "Fast delivery and outstanding customer service. Would buy again!" },
    { name: "Emily Davis", rating: "⭐⭐⭐", review: "Decent experience. Some things could be improved, but overall not bad." },
    { name: "Robert Wilson", rating: "⭐⭐", review: "Not great. Had issues with delivery and support was slow to respond." },
    { name: "Sarah Thompson", rating: "⭐", review: "Terrible experience! Would not recommend. Customer service was unhelpful." }
];

const ReviewSlider = () => {
    const [index, setIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setIndex(prevIndex => (prevIndex === reviews.length - 1 ? 0 : prevIndex + 1));
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const prevReview = () => setIndex(index === 0 ? reviews.length - 1 : index - 1);
    const nextReview = () => setIndex(index === reviews.length - 1 ? 0 : index + 1);

    return (
        <Paper sx={{ p: 3, textAlign: 'center', mt: 3, boxShadow: 3 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>What Customers Say</Typography>
            <motion.div
                key={index}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.5 }}
            >
                <Typography variant="h6">{reviews[index].name}</Typography>
                <Typography sx={{ color: 'gold', fontSize: '1.2rem' }}>{reviews[index].rating}</Typography>
                <Typography variant="body1" sx={{ mt: 1 }}>{reviews[index].review}</Typography>
            </motion.div>
            <div style={{ marginTop: '10px' }}>
                <Button onClick={prevReview} sx={{ mx: 1 }}><ChevronLeft /></Button>
                <Button onClick={nextReview} sx={{ mx: 1 }}><ChevronRight /></Button>
            </div>
        </Paper>
    );
};

const Home = () => {
    return (
        <Container maxWidth="lg">
            {/* Centered Page Title */}
            <Box
                sx={{
                    textAlign: "center",
                    padding: "20px 0",
                }}
            >
                <Typography variant="h4" fontWeight="bold" color="primary">
                    HeartSpeak AI
                </Typography>
            </Box>

            {/* Review Slider Section */}
            <ReviewSlider />

            {/* Page Content */}
            <Grid container spacing={3} sx={{ mt: 2 }}>
                {/* Alerts & Performance */}
                <Grid item xs={6}>
                    <Paper sx={{ p: 2, height: '150px' }}>
                        <Typography variant="h6">Alerts / Performance Metrics</Typography>
                        <Typography variant="body2">Placeholder for alerts and key performance insights.</Typography>
                    </Paper>
                </Grid>

                {/* Dataset Map */}
                <Grid item xs={6}>
                    <Paper sx={{ p: 2, height: '150px' }}>
                        <Typography variant="h6">Dataset vs Dataset Map</Typography>
                        <Typography variant="body2">Map visualization placeholder.</Typography>
                    </Paper>
                </Grid>
            </Grid>

            {/* Default View Info */}
            <Paper sx={{ mt: 3, p: 2, textAlign: 'center' }}>
                <Typography variant="h6">Default View:</Typography>
                <Typography variant="body1">Powered by Azure Power BI (Embedded)</Typography>
            </Paper>
        </Container>
    );
};

export default Home;
