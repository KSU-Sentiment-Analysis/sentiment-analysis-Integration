import React from "react";
import heartspeakIcon from "../assets/heartspeak-icon.png";
import icons8DayAndNight501 from "../assets/icons8-day-and-night-50-1.png";
import icons8HighRisk501 from "../assets/icons8-high-risk-50-1.png";
import icons8Map501 from "../assets/icons8-map-50-1.png";
import icons8Quote4812 from "../assets/icons8-quote-48-1-2.png";
import icons8Quote481 from "../assets/icons8-quote-48-1.png";
import image from "../assets/image.png";
import '../styles/style.css';
import untitledDesign12 from "../assets/untitled-design-1-2.png";
import untitledDesign13 from "../assets/untitled-design-1-3.png";
import untitledDesign1 from "../assets/untitled-design-1.png";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";

const reviews = [
    {
        text: "Heartspeak AI helped us identify user sentiment in real-time. Amazing tool!",
        name: "Jasmine Patel",
        role: "Product Manager",
        quote: icons8Quote481,
        image: untitledDesign1,
      },
      {
        text: "A user-friendly interface and insightful analytics. Highly recommend.",
        name: "Carlos Rivera",
        role: "UX Designer",
        quote: icons8Quote4812,
        image: untitledDesign13,
      },
      {
        text: "The sentiment tracking feature gave us a new level of customer insight.",
        name: "Elena Smith",
        role: "Marketing Director",
        quote: image,
        image: untitledDesign12,
      },
      {
        text: "Our feedback loop has become much faster and more accurate since using Heartspeak.",
        name: "Daniel Kim",
        role: "Customer Experience Lead",
        quote: icons8Quote481,
        image: untitledDesign1,
      },
      {
        text: "Exceptional support and great results. It transformed our communication strategy.",
        name: "Leila Zhang",
        role: "Operations Manager",
        quote: icons8Quote4812,
        image: untitledDesign13,
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
                style={{
                  height: 28,
                  opacity: 0.9,
                }}
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
                    </div>

                    <div className="dataset-vs-dataset">
                        <div className="dataset-vs-dataset-2">Dataset vs Dataset Map</div>
                        <p className="dataset-vs-dataset-3">
                            Visual comparisons of datasets go here.
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
