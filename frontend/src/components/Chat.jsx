// src/components/Chat.jsx
import React, { useState, useEffect, useRef } from "react";
import { chatWithGrok } from "../api";
import axios from "axios";
import "./components.css";
import { v4 as uuidv4 } from "uuid";
import QuickQueryButtons from "./QuickQueryButtons";
import TypingEffect from "./TypingEffect";

const Chat = ({ userId, userLocation }) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [showQuickQueries, setShowQuickQueries] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [showLocationNotice, setShowLocationNotice] = useState(true);
  const [sessionId] = useState(() => `session_${uuidv4()}_${Date.now()}`);
  
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const messageListRef = useRef(null);
  
  // Define quick access buttons
  const quickButtons = [
    { text: "Weather", icon: "🌤️" },
    { text: "Crop Guide", icon: "🌱" },
    { text: "Market Prices", icon: "💰" },
    { text: "Pest Control", icon: "🐛" },
  ];
  
  // Always start with a welcome message on component mount
  useEffect(() => {
    // Always start with a fresh welcome message
    setMessages([
      { 
        role: "assistant", 
        text: "Hello! I'm AgriSense, your agricultural assistant. How can I help with your farming today? You can ask me about weather forecasts, crop recommendations, pest management, market prices, or any other farming concerns." 
      }
    ]);
  }, []);
  
  // Function to scroll to bottom of chat
  const scrollToBottom = (behavior = "smooth") => {
    chatEndRef.current?.scrollIntoView({ behavior });
  };
  
  // Auto-scroll to bottom of chat when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Add message sound effect
  const playMessageSound = (type) => {
    try {
      const sound = new Audio(type === 'sent' ? '/message-sent.mp3' : '/message-received.mp3');
      sound.volume = 0.2;
      sound.play();
    } catch (error) {
      // Silently fail if audio cannot be played
    }
  };
  
  const handleSend = async () => {
    if (!input.trim()) return;
    
    // Add user message to chat
    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: "user", text: userMessage }]);
    setInput("");
    
    // Hide quick queries when sending a message
    setShowQuickQueries(false);
    
    // Play sound effect for sent message
    playMessageSound('sent');
    
    // Scroll to bottom immediately after sending message
    setTimeout(() => scrollToBottom(), 50);
    
    // Show thinking indicator
    setIsThinking(true);
    
    setTimeout(() => {
      setIsThinking(false);
      // Show typing indicator after "thinking" is done
      setIsTyping(true);
    }, 1000);
    
    try {
      // Send message to backend
      const reply = await chatWithGrok(
        userMessage, 
        userLocation.lat, 
        userLocation.lon, 
        `${userId}_${sessionId}`
      );
      
      // Add AI response to chat
      setMessages(prev => [...prev, { role: "assistant", text: reply }]);
      
      // Play sound effect for received message
      playMessageSound('received');
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages(prev => [...prev, { 
        role: "assistant", 
        text: "Sorry, I encountered an error processing your request. Please check your internet connection and try again." 
      }]);
    } finally {
      setIsTyping(false);
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  const handleQuickQuery = (query) => {
    setInput(query);
    setShowQuickQueries(false);
    // Focus on input after selecting a query
    inputRef.current?.focus();
  };
  
  const toggleQuickQueries = () => {
    setShowQuickQueries(prev => !prev);
  };
  
  // Function to get appropriate query based on button type
  
  // Function to handle quick button clicks
  const handleQuickButtonClick = (text) => {
    let query = "";
    switch(text) {
      case "Weather":
        query = "What's the weather forecast for my farm this week?";
        break;
      case "Crop Guide":
        query = "Which crops are best suited for planting now in my region?";
        break;
      case "Market Prices":
        query = "What are the current market prices for common crops?";
        break;
      case "Pest Control":
        query = "How can I control common pests naturally?";
        break;
      default:
        query = text;
    }
    setInput(query);
    inputRef.current?.focus();
  };

  return (
    <div className="chat-container">
      {showLocationNotice && (
        <>
          {userLocation.status === "loading" && (
            <div className="location-notice">
              <p>📍 Getting your location for personalized farming advice...</p>
              <button className="notice-close" onClick={() => setShowLocationNotice(false)}>&times;</button>
            </div>
          )}
          
          {userLocation.status === "error" && (
            <div className="location-notice location-error">
              <p>⚠️ Location access denied. Using default location for generic advice.</p>
              <button className="notice-close" onClick={() => setShowLocationNotice(false)}>&times;</button>
            </div>
          )}
          
          {userLocation.status === "success" && (
            <div className="location-notice location-success">
              <p>📍 Using your location for personalized farming advice</p>
              <button className="notice-close" onClick={() => setShowLocationNotice(false)}>&times;</button>
            </div>
          )}
        </>
      )}
      
      <div className="chat-messages" ref={messageListRef}>
        {messages.map((m, idx) => (
          <div 
            key={idx} 
            className={`message ${m.role === "user" ? "user-message" : "ai-message"}`}
          >
            <div className="message-avatar">
              {m.role === "user" ? "👨‍🌾" : "🌱"}
            </div>
            <div className="message-content">
              {m.role === "assistant" && idx === messages.length - 1 ? (
                <TypingEffect text={m.text} speed={20} />
              ) : (
                m.text.split("\n").map((line, i) => (
                  <p key={i}>{line || <br/>}</p>
                ))
              )}
            </div>
          </div>
        ))}
        
        {isThinking && (
          <div className="message ai-message">
            <div className="message-avatar">🌱</div>
            <div className="message-content thinking-indicator">
              <span>Analyzing farming data...</span>
            </div>
          </div>
        )}
        
        {isTyping && !isThinking && (
          <div className="message ai-message">
            <div className="message-avatar">🌱</div>
            <div className="message-content typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <div ref={chatEndRef} />
      </div>
      
      {showQuickQueries && (
        <div className="extended-quick-queries">
          <QuickQueryButtons onQuerySelect={handleQuickQuery} />
        </div>
      )}
      
      <div className="quick-action-buttons">
        {quickButtons.map((button, idx) => (
          <button 
            key={idx} 
            className="quick-action-btn"
            onClick={() => handleQuickButtonClick(button.text)}
            title={button.text}
          >
            <span className="btn-icon">{button.icon}</span>
            <span className="btn-text">{button.text}</span>
          </button>
        ))}
        
        <button 
          className={`quick-action-btn ${showQuickQueries ? 'active' : ''}`}
          onClick={toggleQuickQueries}
          title="Show more question suggestions"
        >
          <span className="btn-icon">💡</span>
          <span className="btn-text">More Questions</span>
        </button>
      </div>
      
      <div className="chat-input-container">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about crops, weather, or farming advice..."
          rows="2"
          className="chat-input"
        />
        <button 
          onClick={handleSend} 
          disabled={isTyping || isThinking || !input.trim()}
          className="send-button"
          title="Send message"
        >
          {isThinking ? "🧠" : isTyping ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default Chat;
