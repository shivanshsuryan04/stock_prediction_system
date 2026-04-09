"use client";

import { useState, useEffect, useRef, useCallback } from "react";

export interface IndexQuote {
  symbol: string;
  label: string;
  price: number | null;
  change: number | null;
  changePercent: number | null;
  isLoading: boolean;
  isUp: boolean;
}

const REFRESH_MS = 60_000; // refresh every 60 s

export function useMarketIndices() {
  const [nifty, setNifty] = useState<IndexQuote>({
    symbol: "^NSEI", label: "NIFTY 50",
    price: null, change: null, changePercent: null, isLoading: true, isUp: true,
  });
  
  const [sensex, setSensex] = useState<IndexQuote>({
    symbol: "^BSESN", label: "SENSEX",
    price: null, change: null, changePercent: null, isLoading: true, isUp: true,
  });

  // 1. Added Bank Nifty state
  const [bankNifty, setBankNifty] = useState<IndexQuote>({
    symbol: "^NSEBANK", label: "NIFTY BANK",
    price: null, change: null, changePercent: null, isLoading: true, isUp: true,
  });

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchQuote = useCallback(async (symbol: string) => {
    try {
      // Yahoo Finance v8 quoteSummary – no API key required, CORS-safe via Next.js route
      const res = await fetch(`/api/?symbol=${encodeURIComponent(symbol)}`);
      if (!res.ok) throw new Error("bad response");
      const json = await res.json() as {
        price: number; change: number; changePercent: number;
      };
      return json;
    } catch {
      return null;
    }
  }, []);

  const refresh = useCallback(async () => {
    // 2. Added ^NSEBANK to the concurrent fetches
    const [n, s, b] = await Promise.all([
      fetchQuote("^NSEI"),
      fetchQuote("^BSESN"),
      fetchQuote("^NSEBANK"),
    ]);
    
    if (n) setNifty(prev => ({ ...prev, ...n, isLoading: false, isUp: n.change >= 0 }));
    else    setNifty(prev => ({ ...prev, isLoading: false }));

    if (s) setSensex(prev => ({ ...prev, ...s, isLoading: false, isUp: s.change >= 0 }));
    else    setSensex(prev => ({ ...prev, isLoading: false }));

    // 3. Update the Bank Nifty state with the fetched result
    if (b) setBankNifty(prev => ({ ...prev, ...b, isLoading: false, isUp: b.change >= 0 }));
    else    setBankNifty(prev => ({ ...prev, isLoading: false }));
  }, [fetchQuote]);

  useEffect(() => {
    void refresh();
    timerRef.current = setInterval(() => void refresh(), REFRESH_MS);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [refresh]);

  // 4. Return bankNifty so your components can use it
  return { nifty, sensex, bankNifty, refresh };
}