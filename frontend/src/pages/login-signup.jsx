import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate, Routes, Route } from "react-router-dom";

const API_URL = "http://127.0.0.1:8000";
const GOOGLE_CLIENT_ID =
  "941016266537-e2pfksi31c97959fmfin5hgk9iffa40p.apps.googleusercontent.com";
const GOOGLE_REDIRECT_URI = "http://localhost:5173/oauth-callback";
const GOOGLE_SCOPE = "openid email profile";

function getGoogleOAuthURL() {
  const params = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    redirect_uri: GOOGLE_REDIRECT_URI,
    response_type: "token",
    scope: GOOGLE_SCOPE,
    prompt: "select_account",
  });
  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
}

const cardStyle = {
  maxWidth: 400,
  margin: "48px auto",
  padding: "32px 28px 24px 28px",
  background: "#fff",
  borderRadius: 14,
  boxShadow: "0 4px 24px #0002",
  display: "flex",
  flexDirection: "column",
  gap: 18,
};

const inputStyle = {
  padding: "12px",
  borderRadius: 8,
  border: "1px solid #d0d0d0",
  fontSize: "1rem",
  outline: "none",
  marginBottom: 4,
  background: "#fafbfc",
  transition: "border 0.2s",
};

const buttonStyle = {
  padding: "12px 0",
  borderRadius: 8,
  border: "none",
  background: "linear-gradient(90deg, #80e085 0%, #4fda56 100%)",
  color: "#fff",
  fontWeight: 600,
  fontSize: "1rem",
  cursor: "pointer",
  marginTop: 8,
  transition: "background 0.2s",
};

const googleBtnStyle = {
  background: "#fff",
  border: "1px solid #ccc",
  padding: "10px 0",
  borderRadius: 8,
  cursor: "pointer",
  width: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontWeight: 500,
  fontSize: "1rem",
  marginTop: 8,
  boxShadow: "0 1px 4px #0001",
  transition: "border 0.2s",
};

const linkStyle = {
  color: "#1976d2",
  textDecoration: "none",
  fontWeight: 500,
};

const dividerStyle = {
  textAlign: "center",
  margin: "18px 0 8px 0",
  color: "#888",
  fontSize: "0.97rem",
};

export function Login() {
  const [form, setForm] = useState({ gmail: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("gmail", form.gmail);
      formData.append("password", form.password);

      const res = await axios.post(`${API_URL}/login`, formData);
      setMessage(res.data.message);
      if (res.data.token) {
        localStorage.setItem("token", res.data.token);
        window.dispatchEvent(new Event("storage"));
        navigate("/"); // Redirect to main page
      }
    } catch (err) {
      setMessage(
        err.response?.data?.detail || "Login failed. Please try again."
      );
    }
  };

  const handleOAuth = () => {
    window.location.href = getGoogleOAuthURL();
  };

  return (
    <div style={cardStyle}>
      <h2
        style={{
          textAlign: "center",
          marginBottom: 10,
          fontWeight: 700,
          background: "linear-gradient(90deg, #80e085 0%, #4fda56 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          color: "transparent",
        }}
      >
        Login
      </h2>{" "}
      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", flexDirection: "column", gap: 10 }}
      >
        <input
          type="email"
          name="gmail"
          placeholder="Gmail"
          value={form.gmail}
          onChange={handleChange}
          required
          style={inputStyle}
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={form.password}
          onChange={handleChange}
          required
          style={inputStyle}
        />
        <button type="submit" style={buttonStyle}>
          Login
        </button>
      </form>
      {message && (
        <div
          style={{
            color: message.toLowerCase().includes("success")
              ? "#388e3c"
              : "#d32f2f",
            marginTop: 8,
            textAlign: "center",
            fontWeight: 500,
          }}
        >
          {message}
        </div>
      )}
      <div style={dividerStyle}>
        Don't have an account?{" "}
        <a href="/signup" style={linkStyle}>
          Sign Up
        </a>
      </div>
      <div style={dividerStyle}>Or continue with</div>
      <button onClick={handleOAuth} style={googleBtnStyle}>
        <img
          src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png"
          alt="Google"
          style={{ width: 22, verticalAlign: "middle", marginRight: 10 }}
        />
        Google
      </button>
    </div>
  );
}

export function Signup() {
  const [form, setForm] = useState({ gmail: "", name: "", password: "" });
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("gmail", form.gmail);
      formData.append("name", form.name);
      formData.append("password", form.password);

      const res = await axios.post(`${API_URL}/signup`, formData);
      setMessage(res.data.message);
      if (res.data.success) {
        // On successful signup, redirect to login page
        navigate("/login");
      }
    } catch (err) {
      setMessage(
        err.response?.data?.detail || "Signup failed. Please try again."
      );
    }
  };

  const handleOAuth = () => {
    window.location.href = getGoogleOAuthURL();
  };

  return (
    <div style={cardStyle}>
      <h2
        style={{
          textAlign: "center",
          marginBottom: 10,
          fontWeight: 700,
          background: "linear-gradient(90deg, #80e085 0%, #4fda56 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          color: "transparent",
        }}
      >
        Sign Up
      </h2>
      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", flexDirection: "column", gap: 10 }}
      >
        <input
          type="email"
          name="gmail"
          placeholder="Gmail"
          value={form.gmail}
          onChange={handleChange}
          required
          style={inputStyle}
        />
        <input
          type="text"
          name="name"
          placeholder="Name"
          value={form.name}
          onChange={handleChange}
          required
          style={inputStyle}
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={form.password}
          onChange={handleChange}
          required
          style={inputStyle}
        />
        <button type="submit" style={buttonStyle}>
          Sign Up
        </button>
      </form>
      {message && (
        <div
          style={{
            color: message.toLowerCase().includes("success")
              ? "#388e3c"
              : "#d32f2f",
            marginTop: 8,
            textAlign: "center",
            fontWeight: 500,
          }}
        >
          {message}
        </div>
      )}
      <div style={dividerStyle}>
        Already have an account?{" "}
        <a href="/login" style={linkStyle}>
          Login
        </a>
      </div>
      <div style={dividerStyle}>Or continue with</div>
      <button onClick={handleOAuth} style={googleBtnStyle}>
        <img
          src="https://upload.wikimedia.org/wikipedia/commons/4/4a/Logo_2013_Google.png"
          alt="Google"
          style={{ width: 22, verticalAlign: "middle", marginRight: 10 }}
        />
        Google
      </button>
    </div>
  );
}

// --- OAuth Callback Handler ---
export function OAuthCallback() {
  const navigate = useNavigate();

  useEffect(() => {
    // Get access_token from URL hash
    const hash = window.location.hash;
    const params = new URLSearchParams(hash.replace("#", "?"));
    const access_token = params.get("access_token");

    if (access_token) {
      // Fetch user info from Google
      fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
        headers: { Authorization: `Bearer ${access_token}` },
      })
        .then((res) => res.json())
        .then(async (profile) => {
          // Send profile info to backend /oauth endpoint
          const formData = new FormData();
          formData.append("gmail", profile.email);
          formData.append("name", profile.name || profile.email.split("@")[0]);
          formData.append("oauth_provider", "Google");
          formData.append("oauth_id", profile.sub);

          const res = await axios.post(`${API_URL}/oauth`, formData);
          if (res.data.token) {
            localStorage.setItem("token", res.data.token);
            window.dispatchEvent(new Event("storage"));
            navigate("/");
          } else {
            alert("OAuth failed. Please try again.");
            navigate("/login");
          }
        })
        .catch(() => {
          alert("OAuth failed. Please try again.");
          navigate("/login");
        });
    } else {
      alert("OAuth failed. Please try again.");
      navigate("/login");
    }
  }, [navigate]);

  return (
    <div
      style={{
        ...cardStyle,
        textAlign: "center",
        fontSize: "1.1rem",
        color: "#1976d2",
      }}
    >
      Logging you in...
    </div>
  );
}

// --- Export all routes for use in App.jsx ---
export default function AuthRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/oauth-callback" element={<OAuthCallback />} />
    </Routes>
  );
}
