import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import pool from "./db";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Backend running with TypeScript");
});

app.get("/test-db", async (req, res) => {
  try {
    const result = await pool.query("SELECT NOW()");
    res.json(result.rows);
  } catch (error) {
    console.error(error);
    res.status(500).send("Database connection failed");
  }
});

app.get("/health", (req, res) => {
  res.json({ status: "OK", service: "Stock Prediction Backend" });
});

app.listen(5000, () => {
  console.log("Server running at http://localhost:5000");
});
