import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./app/styles/tokens.css";
import "./app/styles/globals.css";
import "./pages/chat-page.css";
import "./pages/chat-mobile.css";
import "./app/styles/zhixueda.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
