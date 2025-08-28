import React, { useState, useRef, useEffect } from "react";
import { FaBars, FaPlus, FaUserCircle, FaSignInAlt } from "react-icons/fa";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

const Sidebar = ({ open, onToggle, user, chats, activeChatId, setActiveChatId, refreshChats, onLogout }) => {
  const [showPopup, setShowPopup] = useState(false);
  const popupRef = useRef();


  // Hide popup when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (popupRef.current && !popupRef.current.contains(event.target)) {
        setShowPopup(false);
      }
    }
    if (showPopup) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showPopup]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    setShowPopup(false);
    window.dispatchEvent(new Event("storage"));
    // Do NOT call setUser here; parent will handle user state.
  };

  const handleLogin = () => {
    window.location.href = "/login";
  };

  const handleDashboard = () => {
    window.location.href = "/dashboard";
  };

  const handleNewChat = async () => {
    console.log(user, user.user_id);
    if (user && user.user_id) {
      const formData = new FormData();
      formData.append("user_id", user.user_id);
      const res = await axios.post(`${API_URL}/user/new_chat`, formData);
      await refreshChats(); // Refresh the chat list first
      setActiveChatId(res.data.chat_id); // Then select the new chat
    }
  };

  const handleSendMessage = async (message) => {
    if (activeChatId) {
      // Send message to the server
      await axios.post(`${API_URL}/chat/${activeChatId}/message`, { text: message });

      // After sending the first message in a chat
      await axios.post(`${API_URL}/user/update_chat_title`, {
        chat_id: activeChatId,
        title: message
      });
    }
  };

  return (
    <>
      {/* Sidebar Toggle Button */}
      <button
        className="sidebar-toggle"
        style={{
          position: "fixed",
          top: 20,
          left: open ? 220 : 20,
          zIndex: 1001,
          background: "#fff",
          border: "1px solid #ccc",
          borderRadius: "50%",
          width: 40,
          height: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
          transition: "left 0.2s",
          cursor: "pointer",
        }}
        onClick={onToggle}
        aria-label="Toggle sidebar"
      >
        <FaBars />
      </button>

      {/* Sidebar */}
      <div
        className="sidebar"
        style={{
          position: "fixed",
          top: 0,
          left: open ? 0 : -220,
          width: 220,
          height: "100vh",
          background: "#f7f7f7",
          borderRight: "1px solid #e0e0e0",
          boxShadow: open ? "2px 0 8px rgba(0,0,0,0.04)" : "none",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between", // <-- keep this
          transition: "left 0.2s",
          zIndex: 1000,
        }}
      >
        <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "24px 16px", fontWeight: "bold", fontSize: 20 }}>
KishanMitra
          </div>
          <button
            onClick={handleNewChat}
            style={{
              margin: "16px",
              padding: "10px 16px",
              background:  "linear-gradient(90deg, #80e085 0%, #4fda56 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              fontSize: 16,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <FaPlus /> New Chat
          </button>
          {/* Make chat list scrollable */}
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
            <ul style={{ paddingLeft: 0, listStyleType: "none", margin: "0 16px" }}>
              {chats.map((chat) => (
                <li
                  key={chat.chat_id}
                  onClick={() => setActiveChatId(chat.chat_id)}
                  style={{
                    padding: "10px",
                    borderRadius: 4,
                    cursor: "pointer",
                    background: activeChatId === chat.chat_id ? "#e3f2fd" : "transparent",
                    transition: "background 0.2s",
                  }}
                >
                  {chat.title || "New Chat"}
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div style={{
    padding: "24px 0",
    textAlign: "center",
    position: "relative",
    background: "#f7f7f7",
  }}>
          {user ? (
            <>
              <div
                style={{ cursor: "pointer", display: "inline-block" }}
                onClick={() => setShowPopup((prev) => !prev)}
              >
                <FaUserCircle size={36} color="#4fda56" />
                <div style={{ fontSize: 14, marginTop: 6, color: "#4fda56", fontWeight: 600 }}>{user.name}</div>
              </div>
              {showPopup && (
                <div
                  ref={popupRef}
                  style={{
                    position: "absolute",
                    bottom: 60,
                    left: "50%",
                    transform: "translateX(-50%)",
                    background: "#fff",
                    border: "1px solid #ccc",
                    borderRadius: 8,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
                    padding: 12,
                    minWidth: 120,
                    zIndex: 2000,
                  }}
                >
                  <button
                    onClick={handleDashboard}
                    style={{
                      width: "100%",
                      padding: "6px 0",
                      background: "none",
                      border: "none",
                      textAlign: "left",
                      cursor: "pointer",
                      fontSize: 15,
                    }}
                  >
                    Dashboard
                  </button>
                  <button
                     onClick={() => {
                      setShowPopup(false);
                      onLogout(); // <-- Call parent logout handler
                    }}
                    style={{
                      width: "100%",
                      padding: "6px 0",
                      background: "none",
                      border: "none",
                      textAlign: "left",
                      cursor: "pointer",
                      color: "#d32f2f",
                      fontSize: 15,
                    }}
                  >
                    Logout
                  </button>
                </div>
              )}
            </>
          ) : (
            <button
              onClick={handleLogin}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "#80e085",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center", // <-- add this
                margin: "0 auto",         // <-- add this
                width: "100%",            // <-- add this for full width centering
              }}
            >
              <FaSignInAlt size={36} />
              <div style={{ fontSize: 14, marginTop: 6 }}>Login</div>
            </button>
          )}
        </div>
      </div>
    </>
  );
};

export default Sidebar;