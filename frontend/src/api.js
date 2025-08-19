// src/api.js
import axios from "axios";

const API_URL = "http://127.0.0.1:8000"; // FastAPI backend

// Chat with AgriSense AI
export const chatWithGrok = async (prompt, lat, lon, userId) => {
  try {
    // Use userId with sessionId component or add timestamp to ensure unique ID
    const formData = new URLSearchParams({
      user_id: userId || `default_user_${Date.now()}`, 
      query: prompt,
      lat: lat || 28.6139, // Default to New Delhi if location not available
      lon: lon || 77.2090,
      k: 5
    });
    
    const res = await axios.post(
      `${API_URL}/ask`,
      formData, 
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );
    return res.data.response || res.data.error;
  } catch (err) {
    console.error("Error in chatWithGrok:", err);
    return "⚠️ Failed to connect to AgriSense backend. Please check your internet connection and try again.";
  }
};

// Ingest a file
export const ingestFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await axios.post(`${API_URL}/ingest/file`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  } catch (err) {
    console.error("Error ingesting file:", err);
    return { error: "File upload failed" };
  }
};

// Retrieve dummy results
export const retrieveResults = async (query, k = 5) => {
  try {
    const res = await axios.post(
      `${API_URL}/retrieve`,
      new URLSearchParams({ query, k })
    );
    return res.data;
  } catch (err) {
    console.error("Error retrieving results:", err);
    return { error: "Retrieve failed" };
  }
};
