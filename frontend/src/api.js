// // src/api.js
// import axios from "axios";

// const API_URL = "http://127.0.0.1:8000"; // FastAPI backend

// // Chat with Grok
// export const chatWithGrok = async (prompt) => {
//   try {
//     const res = await axios.post(
//       `${API_URL}/chat`,
//       new URLSearchParams({ prompt }) // backend expects form data
//     );
//     return res.data.response || res.data.error;
//   } catch (err) {
//     console.error("Error in chatWithGrok:", err);
//     return "⚠️ Failed to connect to backend.";
//   }
// };

// // Ingest a file
// export const ingestFile = async (file) => {
//   const formData = new FormData();
//   formData.append("file", file);

//   try {
//     const res = await axios.post(`${API_URL}/ingest/file`, formData, {
//       headers: { "Content-Type": "multipart/form-data" },
//     });
//     return res.data;
//   } catch (err) {
//     console.error("Error ingesting file:", err);
//     return { error: "File upload failed" };
//   }
// };

// // Retrieve dummy results
// export const retrieveResults = async (query, k = 5) => {
//   try {
//     const res = await axios.post(
//       `${API_URL}/retrieve`,
//       new URLSearchParams({ query, k })
//     );
//     return res.data;
//   } catch (err) {
//     console.error("Error retrieving results:", err);
//     return { error: "Retrieve failed" };
//   }
// };
