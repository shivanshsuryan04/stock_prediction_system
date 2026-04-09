"use client";

import { useState, useEffect, useCallback } from "react";
import { watchlistApi } from "@/lib/api";
import type { WatchlistItem } from "@/types";
// 1. Import your auth store
import { useAuthStore } from "@/lib/auth.store";

interface UseWatchlistReturn {
  items: WatchlistItem[];
  tickerSet: Set<string>;
  isLoading: boolean;
  error: string | null;
  addTicker: (ticker: string) => Promise<void>;
  removeTicker: (ticker: string) => Promise<void>;
  isOnWatchlist: (ticker: string) => boolean;
  refresh: () => void;
}

export function useWatchlist(): UseWatchlistReturn {
  // 2. Pull in isLoading and isAuthenticated from your store
  const { isLoading: isAuthLoading, isAuthenticated } = useAuthStore();
  
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchWatchlist = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const { data } = await watchlistApi.get();
      setItems(data.data);
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        "Failed to load watchlist.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    // 3. Guard: Do not fetch if auth is still loading or user is not logged in
    if (isAuthLoading || !isAuthenticated) return;

    void fetchWatchlist();
  }, [fetchWatchlist, isAuthLoading, isAuthenticated]);

  const addTicker = useCallback(async (ticker: string): Promise<void> => {
    const optimisticItem: WatchlistItem = {
      id: `temp-${Date.now()}`,
      userId: "",
      ticker,
      addedAt: new Date().toISOString(),
      prediction: null,
    };
    setItems((prev) => [optimisticItem, ...prev]);

    try {
      await watchlistApi.add(ticker);
      void fetchWatchlist();
    } catch (err: unknown) {
      setItems((prev) => prev.filter((i) => i.ticker !== ticker));
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        "Failed to add to watchlist.";
      setError(message);
    }
  }, [fetchWatchlist]);

  const removeTicker = useCallback(async (ticker: string): Promise<void> => {
    setItems((prev) => prev.filter((i) => i.ticker !== ticker));

    try {
      await watchlistApi.remove(ticker);
    } catch (err: unknown) {
      void fetchWatchlist();
      const message =
        (err as { response?: { data?: { message?: string } } })?.response?.data?.message ??
        "Failed to remove from watchlist.";
      setError(message);
    }
  }, [fetchWatchlist]);

  const tickerSet = new Set(items.map((i) => i.ticker));
  const isOnWatchlist = (ticker: string): boolean => tickerSet.has(ticker);

  const refresh = useCallback(() => {
    // 4. Guard manual refreshes
    if (isAuthLoading || !isAuthenticated) return;
    void fetchWatchlist();
  }, [fetchWatchlist, isAuthLoading, isAuthenticated]);

  return {
    items,
    tickerSet,
    isLoading, // Notice we return the local isLoading for the UI loader
    error,
    addTicker,
    removeTicker,
    isOnWatchlist,
    refresh,
  };
}