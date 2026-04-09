// ============================================================
// AUTH
// ============================================================
export interface User {
  id: string;
  name: string;
  email: string;
  role: "USER" | "ADMIN";
  createdAt?: string;
}

export interface AuthResponse {
  success: boolean;
  accessToken: string;
  user: User;
  message?: string;
}

export interface RefreshResponse {
  success: boolean;
  accessToken: string;
}

// ============================================================
// PREDICTIONS
// ============================================================
export interface Prediction {
  id: string;
  ticker: string;
  xgbSignal: string;
  lstmSignal: string;
  lstmConf: number;
  finalSignal: string;
  cachedAt: string;
  fromCache: boolean;
}

export interface PredictionEntry {
  ticker: string;
  data: Prediction | null;
  error: string | null;
}

export interface PredictionsResponse {
  success: boolean;
  data: PredictionEntry[];
}

export interface SinglePredictionResponse {
  success: boolean;
  data: Prediction;
}

// ============================================================
// WATCHLIST
// ============================================================
export interface WatchlistItem {
  id: string;
  userId: string;
  ticker: string;
  addedAt: string;
  prediction?: Prediction | null;
}

export interface WatchlistResponse {
  success: boolean;
  data: WatchlistItem[];
}

// ============================================================
// SIGNAL HELPERS
// ============================================================
export type SignalType = "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL";

export interface SignalMeta {
  label: string;
  colorClass: string;
  bgStyle: string;
  borderStyle: string;
  dotColor: string;
  direction: "bull" | "bear" | "neutral";
}

export const getSignalMeta = (signal: string): SignalMeta => {
  const s = signal?.toUpperCase() ?? "";

  if (s.includes("STRONG BUY")) {
    return {
      label: "Strong Buy",
      colorClass: "text-emerald-400",
      bgStyle: "rgba(16,185,129,0.1)",
      borderStyle: "rgba(16,185,129,0.2)",
      dotColor: "#34d399",
      direction: "bull",
    };
  }
  if (s.includes("BUY")) {
    return {
      label: "Buy",
      colorClass: "text-emerald-400",
      bgStyle: "rgba(16,185,129,0.08)",
      borderStyle: "rgba(16,185,129,0.15)",
      dotColor: "#34d399",
      direction: "bull",
    };
  }
  if (s.includes("STRONG SELL")) {
    return {
      label: "Strong Sell",
      colorClass: "text-rose-400",
      bgStyle: "rgba(244,63,94,0.1)",
      borderStyle: "rgba(244,63,94,0.2)",
      dotColor: "#fb7185",
      direction: "bear",
    };
  }
  if (s.includes("SELL")) {
    return {
      label: "Sell",
      colorClass: "text-rose-400",
      bgStyle: "rgba(244,63,94,0.08)",
      borderStyle: "rgba(244,63,94,0.15)",
      dotColor: "#fb7185",
      direction: "bear",
    };
  }
  return {
    label: "Hold",
    colorClass: "text-amber-400",
    bgStyle: "rgba(245,158,11,0.08)",
    borderStyle: "rgba(245,158,11,0.15)",
    dotColor: "#fbbf24",
    direction: "neutral",
  };
};

// ============================================================
// API UTILITIES
// ============================================================
export interface ApiError {
  success: false;
  message: string;
  errors?: Array<{ msg: string; path: string }>;
}

export interface ApiSuccess<T> {
  success: true;
  data?: T;
  message?: string;
}