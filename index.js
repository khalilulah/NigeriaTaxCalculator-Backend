import express from "express";
import mongoose from "mongoose";
import { config } from "dotenv";
import { chat } from "./src/controllers/chatController.js";
import cors from "cors";

config();

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Connect to MongoDB
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Routes
app.get("/", (req, res) => {
  res.json({ message: "Nigeria Tax Chatbot API" });
});

app.get("/health", (req, res) => {
  res.status(200).send("OK");
});

app.post("/chat", chat);

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
