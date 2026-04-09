"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { predictionsApi } from "@/lib/api";
import type { PredictionEntry } from "@/types";
// 1. Import your auth store
import { useAuthStore } from "@/lib/auth.store";

interface UsePredictionsReturn {
  predictions: PredictionEntry[];
  isLoading: boolean;
  isRefreshing: boolean;
  lastUpdated: Date | null;
  error: string | null;
  refresh: () => void;
  bullCount: number;
  bearCount: number;
  holdCount: number;
}

const REFRESH_INTERVAL_MS = 5 * 60 * 1_000;

export function usePredictions(): UsePredictionsReturn {
  // 2. Pull in isLoading and isAuthenticated from your store
  const { isLoading: isAuthLoading, isAuthenticated } = useAuthStore();
  
  const [predictions, setPredictions] = useState<PredictionEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchPredictions = useCallback(async (background = false) => {
    background ? setIsRefreshing(true) : setIsLoading(true);
    setError(null);
    try {
      const { data } = await predictionsApi.getAll();
      setPredictions(data.data);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        "Failed to load predictions.";
      setError(message);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    // 3. Guard: Do not fetch if auth is still loading or user is not logged in
    if (isAuthLoading || !isAuthenticated) return;

    void fetchPredictions(false);
    intervalRef.current = setInterval(() => void fetchPredictions(true), REFRESH_INTERVAL_MS);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [fetchPredictions, isAuthLoading, isAuthenticated]);

  const refresh = useCallback(() => {
    // 4. Guard manual refreshes too
    if (isAuthLoading || !isAuthenticated) return;
    void fetchPredictions(false);
  }, [fetchPredictions, isAuthLoading, isAuthenticated]);

  const bullCount = predictions.filter((p) => p.data?.finalSignal?.toUpperCase().includes("BUY")).length;
  const bearCount = predictions.filter((p) => p.data?.finalSignal?.toUpperCase().includes("SELL")).length;
  const holdCount = predictions.filter((p) => p.data?.finalSignal?.toUpperCase().includes("HOLD")).length;

  return { predictions, isLoading, isRefreshing, lastUpdated, error, refresh, bullCount, bearCount, holdCount };
}