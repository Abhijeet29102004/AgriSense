// src/components/ChatBox.jsx
import React, { useState, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import axios from "axios";
import { FaMicrophone, FaRedo } from "react-icons/fa";
import ReactMarkdown from "react-markdown";
import clogo from "../assets/clogo.jpg";

// --- Inject chat styles ---
const chatStyles = `
.chat-window {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 8px #0001;
  padding: 16px;
  min-height: 300px;
  max-height: 400px;
  overflow-y: auto;
}
.message {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 12px;
}
.message.user {
  flex-direction: row-reverse;
}
.message .avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  background: #f0f0f0;
}
.message span {
  background: #f6f6f6;
  border-radius: 8px;
  padding: 8px 12px;
  max-width: 70%;
  word-break: break-word;
  font-size: 1rem;
  display: block;
}
.message.user span {
  background: #e0ffe0;
}
.message.ai span {
  background: #e6f0ff;
}
.regenerate-btn {
  margin-left: 8px;
  background: #f0f0f0;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  padding: 4px 8px;
  font-size: 1rem;
}
.typing-indicator {
  display: flex;
  gap: 3px;
  margin-top: 8px;
}
.typing-indicator span {
  display: block;
  width: 7px;
  height: 7px;
  background: #bbb;
  border-radius: 50%;
  animation: blink 1.2s infinite both;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}
.chat-input {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}
.chat-input input[type="text"] {
  flex: 1;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ccc;
  font-size: 1rem;
}
.mic-btn {
  background: #f6f6f6;
  border: none;
  border-radius: 8px;
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0 12px;
  transition: background 0.2s;
}
.mic-btn.listening {
  background: #e0ffe0;
}
`;

// Inject the styles once
if (typeof window !== "undefined" && !document.getElementById("chat-css")) {
  const style = document.createElement("style");
  style.id = "chat-css";
  style.innerHTML = chatStyles;
  document.head.appendChild(style);
}

const API_URL = "http://127.0.0.1:8000";
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

const ChatBox = forwardRef(({ activeChatId, user, refreshChats, location, lang }, ref) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [listening, setListening] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const chatWindowRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Fetch chat history when activeChatId changes
  useEffect(() => {
    if (activeChatId) {
      axios
        .get(`${API_URL}/user/chat_history`, { params: { chat_id: activeChatId } })
        .then((res) => setMessages(res.data.history || []));
    } else {
      setMessages([]);
    }
  }, [activeChatId]);

  const handleSend = async (customInput) => {
    if (!activeChatId) return; // Prevent sending if no chat id
    const msg = customInput !== undefined ? customInput : input;
    if (!msg.trim() || !activeChatId || !user) return;

    setMessages((prev) => [...prev, { role: "user", content: msg }]);
    setIsLoading(true);

    const formData = new FormData();
    formData.append("user_id", user.user_id);
    formData.append("chat_id", activeChatId);
    formData.append("query", msg);
    formData.append("lat", location?.lat || 0);
    formData.append("lon", location?.lon || 0);
    formData.append("lang", lang);

    try {
      const res = await axios.post(`${API_URL}/ask`, formData);
      if (res.data.response) {
        setMessages((prev) => [...prev, { role: "ai", content: res.data.response }]);
      }
      // If this is the first user message, update chat title
      if (messages.length === 0) {
        const titleForm = new FormData();
        titleForm.append("chat_id", activeChatId);
        titleForm.append("title", msg);
        await axios.post(`${API_URL}/user/update_chat_title`, titleForm);
        if (refreshChats) refreshChats();
      }
    } catch {
      setMessages((prev) => [...prev, { role: "ai", content: "Error fetching response." }]);
    }
    setInput("");
    setIsLoading(false);
  };

  // Voice input handler
  const startListening = () => {
    if (!SpeechRecognition) {
      alert("Speech Recognition not supported in this browser.");
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = lang === "en" ? "en-IN" : `${lang}-IN`;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    let stoppedByUser = false;

    recognition.onstart = () => setListening(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setListening(false);
      setTimeout(() => {
        handleSend(transcript);
      }, 100);
      stoppedByUser = true;
      recognition.stop();
    };

    recognition.onerror = (event) => {
      setListening(false);
      stoppedByUser = true;
      alert("Voice input error: " + event.error);
    };

    recognition.onend = () => {
      setListening(false);
      if (!stoppedByUser) {
        recognition.start();
      }
    };

    stoppedByUser = false;
    recognition.start();
  };

  // Optional: Regenerate last AI answer
  const handleRegenerate = async () => {
    // Find the last user message
    const lastUserMsgIdx = [...messages].reverse().findIndex(m => m.role === "user");
    if (lastUserMsgIdx !== -1) {
      const idxFromStart = messages.length - 1 - lastUserMsgIdx;
      const lastUserMsg = messages[idxFromStart];

      setIsLoading(true);

      const formData = new FormData();
      formData.append("user_id", user.user_id);
      formData.append("chat_id", activeChatId);
      formData.append("query", lastUserMsg.content);
      formData.append("lat", location?.lat || 0);
      formData.append("lon", location?.lon || 0);
      formData.append("lang", lang);

      try {
        const res = await axios.post(`${API_URL}/ask`, formData);
        if (res.data.response) {
          // Remove any AI answers after the last user message
          setMessages(prev => {
            const uptoUser = prev.slice(0, idxFromStart + 1);
            return [
              ...uptoUser,
              { role: "ai", content: res.data.response }
            ];
          });
        }
      } catch {
        setMessages(prev => {
          const uptoUser = prev.slice(0, idxFromStart + 1);
          return [
            ...uptoUser,
            { role: "ai", content: "Error fetching response." }
          ];
        });
      }
      setIsLoading(false);
    }
  };

  // Expose sendPrompt to parent via ref
  useImperativeHandle(ref, () => ({
    sendPrompt: (template) => {
      handleSend(template);
    }
  }));

  return (
    <div className="chat-window" ref={chatWindowRef}>
      {messages.map((msg, idx) => (
        <div key={idx} className={`message ${msg.role}`}>
          <img
            className="avatar"
            src={msg.role === "user" ? clogo : "ai-avatar.png"}
            alt="avatar"
          />
          <span>
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </span>
          {msg.role === "ai" && idx === messages.length - 1 && !isLoading && (
            <button
              className="regenerate-btn"
              onClick={handleRegenerate}
              title="Regenerate answer"
            >
              <FaRedo />
            </button>
          )}
        </div>
      ))}

      {/* Loading indicator always at the bottom */}
      {isLoading && (
        <div className="typing-indicator" style={{ margin: "12px 0 0 56px" }}>
          <span></span>
          <span></span>
          <span></span>
        </div>
      )}

      {/* Placeholder message when no active chat */}
      {!activeChatId && (
        <div style={{ textAlign: "center", margin: "2em 0" }}>
          Setting up your chat...
        </div>
      )}

      <form
        className="chat-input"
        onSubmit={e => {
          e.preventDefault();
          handleSend();
        }}
      >
        <input
          type="text"
          placeholder="Type your question about crops, weather, or farming..."
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="button"
          className={`mic-btn${listening ? " listening" : ""}`}
          onClick={startListening}
          title="Speak your question"
          style={{ marginRight: 8 }}
          disabled={isLoading}
        >
          {listening ? "🎙️..." : "🎙️"}
        </button>
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? "Sending..." : "Send 🌱"}
        </button>
      </form>
    </div>
  );
});

export default ChatBox;
