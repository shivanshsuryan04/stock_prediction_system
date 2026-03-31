import { spawn } from "child_process";
import path from "path";
import prisma from "../config/db";
import { logger } from "../utils/logger";
import { PredictionResult, PythonApiPrediction, PredictionEntry } from "../types";

const PYTHON_API_URL = process.env.PYTHON_ML_API_URL ?? "http://localhost:8000";
const ML_DIR = process.env.ML_DIR ?? path.join(__dirname, "../../../../ml-model");
const CACHE_TTL_MINUTES = 30;

export const SUPPORTED_TICKERS: readonly string[] = [
  "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
  "SBIN.NS", "AXISBANK.NS", "WIPRO.NS", "HCLTECH.NS", "ITC.NS",
  "MARUTI.NS", "BHARTIARTL.NS",
];

// ─── Approach 1: HTTP to FastAPI ────────────────────────────────────────────
const getPredictionFromMicroservice = async (ticker: string): Promise<PythonApiPrediction> => {
  const url = `${PYTHON_API_URL}/predict/${encodeURIComponent(ticker)}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      const body = await response.json().catch(() => ({})) as { detail?: string };
      throw new Error(body.detail ?? `Python API responded with ${response.status}`);
    }
    return (await response.json()) as PythonApiPrediction;
  } finally {
    clearTimeout(timeout);
  }
};

// ─── Approach 2: child_process fallback ─────────────────────────────────────
const getPredictionViaChildProcess = (ticker: string): Promise<PythonApiPrediction> => {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(ML_DIR, "predict_ensemble.py");

    const python = spawn("python", [scriptPath, "--ticker", ticker], {
      cwd: ML_DIR,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data: Buffer) => (stdout += data.toString()));
    python.stderr.on("data", (data: Buffer) => (stderr += data.toString()));

    python.on("close", (code: number | null) => {
      if (code !== 0) {
        logger.error(`Python process failed for ${ticker}: ${stderr}`);
        return reject(new Error("ML model execution failed."));
      }
      try {
        resolve(JSON.parse(stdout.trim()) as PythonApiPrediction);
      } catch {
        reject(new Error("Failed to parse ML model output."));
      }
    });

    python.on("error", (err: Error) => {
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
};

// ─── Main service function ───────────────────────────────────────────────────
export const getPrediction = async (ticker: string): Promise<PredictionResult> => {
  if (!SUPPORTED_TICKERS.includes(ticker)) {
    const err = Object.assign(new Error(`Ticker "${ticker}" is not supported.`), {
      statusCode: 400,
    });
    throw err;
  }

  // Check cache
  const cached = await prisma.predictionCache.findUnique({ where: { ticker } });
  if (cached) {
    const ageMinutes = (Date.now() - cached.cachedAt.getTime()) / 1000 / 60;
    if (ageMinutes < CACHE_TTL_MINUTES) {
      logger.info(`Cache hit for ${ticker} (${Math.round(ageMinutes)}min old)`);
      return { ...cached, fromCache: true };
    }
  }

  logger.info(`Fetching fresh prediction for ${ticker}...`);
  let prediction: PythonApiPrediction;

  try {
    prediction = await getPredictionFromMicroservice(ticker);
  } catch (microserviceError) {
    logger.warn(
      `Microservice unavailable, falling back to child_process: ${(microserviceError as Error).message}`
    );
    prediction = await getPredictionViaChildProcess(ticker);
  }

  const upserted = await prisma.predictionCache.upsert({
    where: { ticker },
    update: {
      xgbSignal: prediction.xgboost_prediction,
      lstmSignal: prediction.lstm_prediction,
      lstmConf: prediction.lstm_confidence,
      finalSignal: prediction.final_signal,
      cachedAt: new Date(),
    },
    create: {
      ticker,
      xgbSignal: prediction.xgboost_prediction,
      lstmSignal: prediction.lstm_prediction,
      lstmConf: prediction.lstm_confidence,
      finalSignal: prediction.final_signal,
    },
  });

  return { ...upserted, fromCache: false };
};

export const getAllPredictions = async (): Promise<PredictionEntry[]> => {
  const results = await Promise.allSettled(
    SUPPORTED_TICKERS.map((ticker) => getPrediction(ticker))
  );

  return results.map((result, i) => ({
    ticker: SUPPORTED_TICKERS[i] as string,
    ...(result.status === "fulfilled"
      ? { data: result.value, error: null }
      : { data: null, error: (result.reason as Error).message }),
  }));
};