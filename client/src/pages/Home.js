import React, { useState, useEffect } from "react";
import heartspeakIcon from "../assets/heartspeak-icon.png";
import icons8DayAndNight501 from "../assets/icons8-day-and-night-50-1.png";
import icons8HighRisk501 from "../assets/icons8-high-risk-50-1.png";
import icons8Map501 from "../assets/icons8-map-50-1.png";
import icons8Quote4812 from "../assets/icons8-quote-48-1-2.png";
import icons8Quote481 from "../assets/icons8-quote-48-1.png";
import image from "../assets/image.png";
import "../styles/style.css";
import untitledDesign12 from "../assets/untitled-design-1-2.png";
import untitledDesign13 from "../assets/untitled-design-1-3.png";
import untitledDesign1 from "../assets/untitled-design-1.png";
import { motion, AnimatePresence } from "framer-motion";
import Papa from "papaparse";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer
} from "recharts";

const COLORS = ['#4CAF50', '#8BC34A', '#FFEB3B', '#FF9800', '#F44336'];
const TARGET_SENTIMENTS = ['very positive', 'positive', 'neutral', 'negative', 'very negative'];

const reviews = [
  {
    text: "This hoodie is the softest thing I’ve ever worn. I never want to take it off.",
    name: "Jasmine Patel",
    role: "Verified Buyer",
    stars: 5,
    quote: icons8Quote481,
    image: untitledDesign1,
  },
  {
    text: "The fit of these jeans is unreal—snug in all the right places without being tight.",
    name: "Carlos Rivera",
    role: "Fashion Enthusiast",
    stars: 4,
    quote: icons8Quote481,
    image: untitledDesign1,
  },
  {
    text: "Bought the oversized blazer for work and it instantly elevated my whole wardrobe.",
    name: "Elena Smith",
    role: "Style Blogger",
    stars: 5,
    quote: icons8Quote481,
    image: untitledDesign1,
  },
  {
    text: "These sneakers are not only stylish but insanely comfortable. I wear them everywhere.",
    name: "Daniel Kim",
    role: "Sneaker Collector",
    stars: 4,
    quote: icons8Quote481,
    image: untitledDesign1,
  },
  {
    text: "The quality of the stitching and fabric is top-notch. This is premium fashion done right.",
    name: "Leila Zhang",
    role: "Luxury Fashion Fan",
    stars: 5,
    quote: icons8Quote481,
    image: untitledDesign1,
  },
];

const ReviewSlider = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % reviews.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const current = reviews[index];

  return (
    <div
      className="review-slider"
      style={{
        width: "100%",
        maxWidth: 1600,
        margin: "0 auto",
        position: "relative"
      }}
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
          style={{
            display: "flex",
            alignItems: "center",
            background: "#fff",
            borderRadius: "10px",
            padding: "24px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            gap: "20px",
          }}
        >
          {/* Left: Image + Quote */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "10px" }}>
            <img
              src={current.quote}
              alt="Quote"
              style={{ height: 28, opacity: 0.9 }}
            />
            <img
              src={current.image}
              alt={current.name}
              style={{
                width: 70,
                height: 70,
                borderRadius: "50%",
                objectFit: "cover",
              }}
            />
          </div>

          {/* Center: Quote Text */}
          <div style={{ flex: 1 }}>
            <p style={{ fontStyle: "italic", fontSize: "1rem", color: "#000", marginBottom: "12px" }}>
              {current.text}
            </p>

          {/* Stars */}
          <div style={{ marginBottom: "8px" }}>
            {Array.from({ length: 5 }, (_, i) => (
              <span key={i} style={{ color: i < current.stars ? "#FFD700" : "#E0E0E0", fontSize: "20px" }}>
                ★
              </span>
            ))}
          </div>
          
            <div style={{ textAlign: "right" }}>
              <div style={{ fontWeight: 600, color: "#000" }}>{current.name}</div>
              <div style={{ fontSize: "0.9rem", color: "#828282" }}>{current.role}</div>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export const Homepage = () => {
  const [sentimentData, setSentimentData] = useState([]);

  useEffect(() => {
    fetch("/data/pre_advanced_sentiment_output.csv")
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (result) => {
            const sentimentCount = {};
            result.data.forEach((row) => {
              const rawSentiment = row.predicted_sentiment || row.sentiment || row.Sentiment || row.label;
              const sentiment = rawSentiment?.trim()?.toLowerCase();
              if (sentiment) {
                sentimentCount[sentiment] = (sentimentCount[sentiment] || 0) + 1;
              }
            });
            const parsedData = TARGET_SENTIMENTS.map((label) => ({
              name: label,
              value: sentimentCount[label] || 0
            }));
            setSentimentData(parsedData);
          },
        });
      });
  }, []);

  return (
    <div className="homepage">
      <div className="div">

        <div className="welcome-header">Welcome to Heartspeak AI</div>

        <p className="welcome-subtitle">
          Analyze emotions, visualize trends, and generate intelligent
          feedback—all in one platform.
        </p>

        <div className="metric-section">
          <div className="alerts-metric">
            <div className="alerts-metric-title">Alerts/Metric</div>
            <p className="p">Real-time sentiment metrics will appear here.</p>
            <img className="high-risk" alt="High risk" src={icons8HighRisk501} />

            {/* Sentiment Pie Chart */}
            {sentimentData.length > 0 && (
              <div style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sentimentData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {sentimentData.map((_, index) => (
                        <Cell key={`sentiment-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <div className="dataset-vs-dataset">
            <div className="dataset-vs-dataset-2">Monthly Comparisons</div>
            <p className="dataset-vs-dataset-3">
              Monthly comparisons of datasets go here.
            </p>
            <img className="map" alt="Map" src={icons8Map501} />
          </div>
        </div>

        <div className="customer-reviews">Customer Reviews</div>

        <div className="review-cards">
          <ReviewSlider />
        </div>

        <p className="footer">© 2025 Heartspeak AI. All rights reserved.</p>
      </div>
    </div>
  );
};
