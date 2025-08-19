

// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [prompt, setPrompt] = useState("");
//   const [chatResponse, setChatResponse] = useState("");

//   const [query, setQuery] = useState("");
//   const [retrievedResponse, setRetrievedResponse] = useState("");
//   const [contextUsed, setContextUsed] = useState("");

//   const [csvFile, setCsvFile] = useState(null);
//   const [csvStatus, setCsvStatus] = useState("");

//   const backendUrl = "http://127.0.0.1:8000"; // Update if your backend runs elsewhere

//   // ───────────── CHAT ─────────────
//   const handleChatSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const formData = new FormData();
//       formData.append("prompt", prompt);

//       const res = await axios.post(`${backendUrl}/chat`, formData);
//       setChatResponse(res.data.response);
//     } catch (err) {
//       console.error(err);
//       setChatResponse("Error sending prompt.");
//     }
//   };

//   // ───────────── RETRIEVE ─────────────
//   const handleRetrieveSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const formData = new FormData();
//       formData.append("query", query);
//       formData.append("k", 5); // number of docs to retrieve

//       const res = await axios.post(`${backendUrl}/retrieve`, formData);
//       setRetrievedResponse(res.data.response);
//       setContextUsed(res.data.context_used);
//     } catch (err) {
//       console.error(err);
//       setRetrievedResponse("Error retrieving info.");
//     }
//   };

//   // ───────────── CSV INGEST ─────────────
//   const handleCsvUpload = async (e) => {
//     e.preventDefault();
//     if (!csvFile) return;

//     const formData = new FormData();
//     formData.append("file", csvFile);

//     try {
//       const res = await axios.post(`${backendUrl}/ingest/file`, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });
//       setCsvStatus(
//         `File uploaded: ${res.data.filename} | Rows ingested: ${res.data.rows_ingested}`
//       );
//     } catch (err) {
//       console.error(err);
//       setCsvStatus("CSV upload failed.");
//     }
//   };

//   return (
//     <div style={{ padding: "20px", fontFamily: "Arial" }}>
//       <h1>Groq + LangChain Frontend</h1>

//       {/* ───────────── CHAT ───────────── */}
//       <section style={{ marginBottom: "30px" }}>
//         <h2>Chat with AI</h2>
//         <form onSubmit={handleChatSubmit}>
//           <input
//             type="text"
//             placeholder="Enter prompt..."
//             value={prompt}
//             onChange={(e) => setPrompt(e.target.value)}
//             style={{ width: "300px", marginRight: "10px" }}
//           />
//           <button type="submit">Send</button>
//         </form>
//         <p><strong>Response:</strong> {chatResponse}</p>
//       </section>

//       {/* ───────────── RETRIEVE ───────────── */}
//       <section style={{ marginBottom: "30px" }}>
//         <h2>Retrieve + AI Answer</h2>
//         <form onSubmit={handleRetrieveSubmit}>
//           <input
//             type="text"
//             placeholder="Enter query..."
//             value={query}
//             onChange={(e) => setQuery(e.target.value)}
//             style={{ width: "300px", marginRight: "10px" }}
//           />
//           <button type="submit">Retrieve</button>
//         </form>
//         <p><strong>Context Used:</strong> {contextUsed}</p>
//         <p><strong>Response:</strong> {retrievedResponse}</p>
//       </section>

//       {/* ───────────── CSV INGEST ───────────── */}
//       <section style={{ marginBottom: "30px" }}>
//         <h2>Upload CSV for Ingestion</h2>
//         <form onSubmit={handleCsvUpload}>
//           <input
//             type="file"
//             accept=".csv"
//             onChange={(e) => setCsvFile(e.target.files[0])}
//             style={{ marginRight: "10px" }}
//           />
//           <button type="submit">Upload</button>
//         </form>
//         <p>{csvStatus}</p>
//       </section>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [csvFile, setCsvFile] = useState(null);
//   const [csvStatus, setCsvStatus] = useState("");

//   const [askQuery, setAskQuery] = useState("");
//   const [askResponse, setAskResponse] = useState("");
//   const [askContext, setAskContext] = useState("");

//   const backendUrl = "http://127.0.0.1:8000"; // Change if needed

//   // ───────────── CSV UPLOAD ─────────────
//   const handleCsvUpload = async (e) => {
//     e.preventDefault();
//     if (!csvFile) return;

//     const formData = new FormData();
//     formData.append("file", csvFile);

//     try {
//       const res = await axios.post(`${backendUrl}/ingest/file`, formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });
//       setCsvStatus(
//         `File uploaded: ${res.data.filename} | Rows ingested: ${res.data.rows_ingested} | Chunks: ${res.data.chunks_created}`
//       );
//     } catch (err) {
//       console.error(err);
//       setCsvStatus("CSV upload failed.");
//     }
//   };

//   // ───────────── ASK AI ─────────────
//   const handleAskSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const formData = new FormData();
//       formData.append("query", askQuery);
//       formData.append("k", 5);

//       const res = await axios.post(`${backendUrl}/ask`, formData);
//       setAskResponse(res.data.response);
//       setAskContext(res.data.context_used);
//     } catch (err) {
//       console.error(err);
//       setAskResponse("Error processing query.");
//     }
//   };

//   return (
//     <div style={{ padding: "20px", fontFamily: "Arial" }}>
//       <h1>Groq + LangChain + CSV AI</h1>

//       {/* ───────────── CSV INGEST ───────────── */}
//       <section style={{ marginBottom: "30px" }}>
//         <h2>Upload CSV for Ingestion</h2>
//         <form onSubmit={handleCsvUpload}>
//           <input
//             type="file"
//             accept=".csv"
//             onChange={(e) => setCsvFile(e.target.files[0])}
//             style={{ marginRight: "10px" }}
//           />
//           <button type="submit">Upload</button>
//         </form>
//         <p>{csvStatus}</p>
//       </section>

//       {/* ───────────── ASK AI ───────────── */}
//       <section style={{ marginBottom: "30px" }}>
//         <h2>Ask AI (with CSV + General fallback)</h2>
//         <form onSubmit={handleAskSubmit}>
//           <input
//             type="text"
//             placeholder="Enter your question..."
//             value={askQuery}
//             onChange={(e) => setAskQuery(e.target.value)}
//             style={{ width: "300px", marginRight: "10px" }}
//           />
//           <button type="submit">Ask</button>
//         </form>
//         <p><strong>Context Used:</strong> {askContext}</p>
//         <p><strong>Response:</strong> {askResponse}</p>
//       </section>
//     </div>
//   );
// }

// export default App;


import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";
import "./App.css";
import Chat from "./components/Chat";
import WeatherWidget from "./components/WeatherWidget";

function App() {
  const [userId] = useState(() => {
    // Generate a new UUID for each session (page load)
    // While keeping a persistent user ID in localStorage
    const storedUserId = localStorage.getItem("agrisense_user_id");
    const sessionUserId = uuidv4(); // New ID for this session
    
    // Store base user ID if not already stored
    if (!storedUserId) {
      localStorage.setItem("agrisense_user_id", sessionUserId);
      return sessionUserId;
    }
    
    // Return stored ID with a unique session identifier
    return `${storedUserId}_${Date.now()}`;
  });
  
  const [darkMode, setDarkMode] = useState(() => {
    // Get user's preference from localStorage or use system preference
    const saved = localStorage.getItem("agrisense_darkmode");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    return saved ? saved === "true" : prefersDark;
  });
  
  const [showWeather, setShowWeather] = useState(false);
  const [userLocation, setUserLocation] = useState({
    lat: null,
    lon: null,
    status: "pending" // pending, loading, success, error
  });
  
  // Dark mode toggle
  useEffect(() => {
    document.body.className = darkMode ? "dark-mode" : "";
    localStorage.setItem("agrisense_darkmode", darkMode);
  }, [darkMode]);
  
  // Handle page refresh/load to ensure fresh chat
  useEffect(() => {
    // Set a flag in sessionStorage to identify this as a new session
    sessionStorage.setItem("agrisense_session_start", Date.now().toString());
    
    // Clear any temporary chat data in sessionStorage
    sessionStorage.removeItem("agrisense_chat_messages");
    
    // Function to handle page refresh or close
    const handleBeforeUnload = () => {
      // This function runs before the page is refreshed/closed
      sessionStorage.removeItem("agrisense_session_start");
    };
    
    // Add event listener
    window.addEventListener("beforeunload", handleBeforeUnload);
    
    // Cleanup
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  // Get user location
  useEffect(() => {
    if (navigator.geolocation) {
      setUserLocation(prev => ({ ...prev, status: "loading" }));
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            status: "success"
          });
        },
        (error) => {
          console.error("Error getting location:", error);
          setUserLocation(prev => ({ 
            ...prev, 
            status: "error",
            lat: 28.6139, // Default fallback to New Delhi
            lon: 77.2090
          }));
        },
        { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
      );
    } else {
      setUserLocation(prev => ({ 
        ...prev, 
        status: "error",
        lat: 28.6139, // Default fallback to New Delhi 
        lon: 77.2090
      }));
    }
  }, []);

  // Toggle weather widget
  const toggleWeatherWidget = () => {
    setShowWeather(prev => !prev);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <img src="/vite.svg" alt="AgriSense Logo" className="logo-image" />
          <h1>AgriSense</h1>
        </div>
        
        <div className="header-controls">
          <button 
            className="weather-toggle" 
            onClick={toggleWeatherWidget} 
            title="Weather Information"
          >
            {showWeather ? "🌤️" : "🌦️"}
          </button>
          
          <button 
            className="dark-toggle" 
            onClick={() => setDarkMode(prev => !prev)}
            title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {darkMode ? "☀️" : "🌙"}
          </button>
        </div>
      </header>

      {showWeather && (
        <WeatherWidget 
          lat={userLocation.lat} 
          lon={userLocation.lon} 
          locationStatus={userLocation.status}
          onClose={toggleWeatherWidget}
        />
      )}

      <main className="app-main">
        <Chat userId={userId} userLocation={userLocation} />
      </main>
      
      <footer className="app-footer">
        <p>© {new Date().getFullYear()} AgriSense - AI-powered farming assistant</p>
      </footer>
    </div>
  );
}

export default App;
