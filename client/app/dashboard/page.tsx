"use client";

import { useState, useEffect, memo } from "react";
import { StockCard } from "@/components/ui/StockCard";
import { ConfidenceBar } from "@/components/charts/ConfidenceBar";
import { usePredictions } from "@/hooks/usePredictions";
import { useWatchlist } from "@/hooks/useWatchlist";
import { ComposedChart, Area, Line, ResponsiveContainer, YAxis, Tooltip, XAxis } from "recharts";

type FilterSignal = "ALL" | "BUY" | "SELL" | "HOLD";

const STAT_CONFIGS = [
  { key: "tracked",  label: "TRACKED",  color: "#06b6d4",  sub: "NSE stocks" },
  { key: "bullish",  label: "BULLISH",  color: "#10b981",  sub: "Buy signals" },
  { key: "bearish",  label: "BEARISH",  color: "#f43f5e",  sub: "Sell signals" },
  { key: "neutral",  label: "NEUTRAL",  color: "#f59e0b",  sub: "Hold signals" },
];

// --- CUSTOM EXPLANATORY TOOLTIP ---
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    // Safe fallback if data points are temporarily undefined
    if (!data) return null;

    const isBullish = data.price > (data.ma10 || 0);

    return (
      <div style={{
        background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.1)",
        padding: "10px 14px", borderRadius: "6px", boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
        minWidth: "160px"
      }}>
        <div style={{ fontSize: 10, color: "#94a3b8", marginBottom: 8, letterSpacing: "0.05em", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: 4 }}>
          TIME: {label}
        </div>
        
        {/* Full Indicator Grid with safe property access */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 12px" }}>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>Price:</div>
          <div style={{ fontSize: 12, fontWeight: 700, color: isBullish ? "#10b981" : "#f43f5e", textAlign: "right" }}>₹{data.price?.toFixed(2) || "0.00"}</div>
          
          <div style={{ fontSize: 11, color: "#94a3b8" }}>MA (10):</div>
          <div style={{ fontSize: 11, color: "#06b6d4", textAlign: "right", fontWeight: 600 }}>₹{data.ma10?.toFixed(2) || "0.00"}</div>
          
          <div style={{ fontSize: 11, color: "#94a3b8" }}>MA (50):</div>
          <div style={{ fontSize: 11, color: "#cbd5e1", textAlign: "right" }}>₹{data.ma50?.toFixed(2) || "0.00"}</div>
          
          <div style={{ fontSize: 11, color: "#94a3b8" }}>RSI (14):</div>
          <div style={{ fontSize: 11, color: (data.rsi || 50) > 70 ? "#f43f5e" : (data.rsi || 50) < 30 ? "#10b981" : "#cbd5e1", textAlign: "right" }}>
            {data.rsi?.toFixed(1) || "50.0"}
          </div>
          
          <div style={{ fontSize: 11, color: "#94a3b8" }}>MACD:</div>
          <div style={{ fontSize: 11, color: (data.macd || 0) > 0 ? "#10b981" : "#f43f5e", textAlign: "right" }}>
            {(data.macd || 0) > 0 ? "+" : ""}{data.macd?.toFixed(2) || "0.00"}
          </div>

          <div style={{ fontSize: 11, color: "#94a3b8" }}>Volume:</div>
          <div style={{ fontSize: 11, color: "#cbd5e1", textAlign: "right" }}>{((data.volume || 0) / 1000).toFixed(1)}k</div>
        </div>
        
        <div style={{
          marginTop: 8, paddingTop: 8, borderTop: "1px solid rgba(255,255,255,0.05)",
          fontSize: 10, color: isBullish ? "#10b981" : "#f43f5e", fontWeight: 500, textAlign: "center"
        }}>
          {isBullish ? "▲ Bullish (Price > MA10)" : "▼ Bearish (Price < MA10)"}
        </div>
      </div>
    );
  }
  return null;
};

// --- HELPER MATH FUNCTIONS ---
const calculateMA = (data: any[], index: number, period: number) => {
  if (index < period - 1) return data[index]?.price || 0;
  let sum = 0;
  for (let i = index - period + 1; i <= index; i++) sum += (data[i]?.price || 0);
  return sum / period;
};

const calculateRSI = (data: any[], index: number, period: number = 14) => {
  if (index < period) return 50;
  let gains = 0, losses = 0;
  for (let i = index - period + 1; i <= index; i++) {
    const diff = (data[i]?.price || 0) - (data[i - 1]?.price || 0);
    if (diff > 0) gains += diff;
    else losses -= diff;
  }
  const avgGain = gains / period;
  const avgLoss = losses / period;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
};

// --- RECHARTS REAL-TIME COMPONENT ---
const RealTimeChart = memo(({ ticker, index = 0 }: { ticker: string, index?: number }) => {
  const [chartData, setChartData] = useState<any[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let intervalId: NodeJS.Timeout;

    const fetchRealData = async () => {
      const targetUrl = encodeURIComponent(`https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?region=IN&interval=1m&range=1d`);
      
      const proxies = [
        `https://api.allorigins.win/raw?url=${targetUrl}&cacheBust=${Date.now()}`,
        `https://api.codetabs.com/v1/proxy/?quest=${decodeURIComponent(targetUrl)}`
      ];

      let data = null;
      let lastError = null;

      for (const proxyUrl of proxies) {
        try {
          const res = await fetch(proxyUrl);
          if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
          data = await res.json();
          break; 
        } catch (err) {
          lastError = err;
        }
      }

      if (!data) throw new Error("All proxies failed to fetch data.");
      
      const result = data?.chart?.result?.[0];
      if (!result || !result.timestamp || !result.indicators?.quote?.[0]?.close) {
         throw new Error("Invalid data format from API");
      }

      const timestamps = result.timestamp || [];
      const quotes = result.indicators.quote[0] || {};
      const closes = quotes.close || [];
      const volumes = quotes.volume || [];

      // 1. Safe extraction into a strict array
      const rawData: any[] = [];
      let prevEma12 = closes[0] || 0;
      let prevEma26 = closes[0] || 0;

      for (let i = 0; i < timestamps.length; i++) {
        const price = closes[i];
        if (price == null || isNaN(price)) continue;

        if (i > 0) {
          prevEma12 = price * (2 / 13) + prevEma12 * (1 - (2 / 13));
          prevEma26 = price * (2 / 27) + prevEma26 * (1 - (2 / 27));
        }

        rawData.push({
          time: new Date(timestamps[i] * 1000).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false }),
          price: price,
          volume: volumes[i] || 0,
          macd: prevEma12 - prevEma26
        });
      }

      if (rawData.length === 0) throw new Error("No valid price data returned");

      // 2. Safe mapping into a new distinct array
      const enrichedData = rawData.map((pt, i, arr) => ({
        ...pt,
        ma10: calculateMA(arr, i, 10),
        ma20: calculateMA(arr, i, 20),
        ma50: calculateMA(arr, i, 50),
        rsi: calculateRSI(arr, i, 14),
      }));

      if (mounted) {
        // 3. Absolute array type guarantee before slicing
        const safeDataForState = Array.isArray(enrichedData) ? enrichedData : [];
        setChartData(safeDataForState.slice(-40));
        setIsLive(true);
        setErrorMsg(null);
      }
    };

    const loadWithRetryAndStagger = async () => {
      const delay = index * 400;
      await new Promise(r => setTimeout(r, delay));

      let success = false;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          if (!mounted) return;
          await fetchRealData();
          success = true;
          break;
        } catch (err) {
          if (!mounted) return;
          console.warn(`Attempt ${attempt} failed for ${ticker}. Retrying...`);
          await new Promise(r => setTimeout(r, 1000 * attempt));
        }
      }

      if (!success && mounted && chartData.length === 0) {
        setErrorMsg("NETWORK BUSY / RETRYING...");
      }
    };

    loadWithRetryAndStagger();

    intervalId = setInterval(() => {
      if (mounted) fetchRealData().catch(() => {});
    }, 30000 + Math.floor(Math.random() * 5000)); 
    
    return () => {
      mounted = false;
      clearInterval(intervalId); 
    };
  }, [ticker, index]);

  if (chartData.length === 0) {
    return (
      <div style={{ height: "240px", width: "100%", background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.05)", borderTop: "none", borderRadius: "0 0 4px 4px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 10, color: errorMsg ? "#f59e0b" : "#475569", letterSpacing: "0.1em", animation: errorMsg ? "none" : "pulse 1.5s infinite" }}>
          {errorMsg || "LOADING LIVE DATA..."}
        </span>
      </div>
    );
  }

  const latest = chartData[chartData.length - 1];
  const isPositive = latest.price >= (latest.ma10 || 0);
  const chartColor = isPositive ? "#10b981" : "#f43f5e";
  
  const prices = chartData.map(d => d.price);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  return (
    <div style={{
      height: "240px", width: "100%", 
      background: "#0a0f1a",
      border: "1px solid rgba(255,255,255,0.05)", borderTop: "none",
      borderRadius: "0 0 4px 4px",
      position: "relative",
      marginTop: "-4px",
      paddingTop: 12
    }}>
      <div style={{ padding: "0 15px", position: "relative", zIndex: 10, display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: chartColor, fontFamily: "inherit", lineHeight: 1.1 }}>
            ₹{latest.price?.toFixed(2) || "0.00"}
          </div>
          <div style={{ fontSize: 9, color: "#94a3b8", display: "flex", alignItems: "center", gap: 5, marginTop: 4 }}>
            <span style={{ display: "inline-block", width: 6, height: 6, borderRadius: "50%", background: isLive ? "#10b981" : "#f59e0b", animation: isLive ? "pulse 2s infinite" : "none" }} />
            {isLive ? "REAL-TIME" : "MARKET CLOSED"}
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 10, fontSize: 9, color: "#64748b", fontWeight: 600 }}>
            <span>RSI: <span style={{ color: (latest.rsi || 50) > 70 ? "#f43f5e" : (latest.rsi || 50) < 30 ? "#10b981" : "#cbd5e1" }}>{latest.rsi?.toFixed(1) || "50.0"}</span></span>
            <span>MACD: <span style={{ color: (latest.macd || 0) > 0 ? "#10b981" : "#f43f5e" }}>{(latest.macd || 0) > 0 ? "+" : ""}{latest.macd?.toFixed(2) || "0.00"}</span></span>
            <span>VOL: <span style={{ color: "#cbd5e1" }}>{((latest.volume || 0) / 1000).toFixed(1)}k</span></span>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
          <div style={{ 
            background: isPositive ? "rgba(16, 185, 129, 0.1)" : "rgba(244, 63, 94, 0.1)",
            border: `1px solid ${isPositive ? "rgba(16, 185, 129, 0.2)" : "rgba(244, 63, 94, 0.2)"}`,
            padding: "3px 8px", borderRadius: "4px", fontSize: 9, color: chartColor, fontWeight: 600, letterSpacing: "0.05em"
          }}>
            TREND: {isPositive ? "BULLISH" : "BEARISH"}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
            <div style={{ fontSize: 9, color: "#94a3b8", display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 8, height: 2, background: chartColor }} /> Price
            </div>
            <div style={{ fontSize: 9, color: "#94a3b8", display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", width: 8, height: 2, background: "#06b6d4" }} /> MA(10)
            </div>
          </div>
        </div>
      </div>

      <div style={{ height: "150px", width: "100%", position: "absolute", bottom: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={`colorPrice_${ticker}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={chartColor} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={chartColor} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis dataKey="time" hide />
            <YAxis domain={[minPrice - (minPrice * 0.001), maxPrice + (maxPrice * 0.001)]} hide />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '4 4' }} />
            <Area 
              type="monotone" 
              dataKey="price" 
              stroke={chartColor} 
              strokeWidth={2}
              fillOpacity={1} 
              fill={`url(#colorPrice_${ticker})`} 
              isAnimationActive={true} 
              animationDuration={800}
            />
            <Line 
              type="monotone" 
              dataKey="ma10" 
              stroke="#06b6d4" 
              strokeWidth={1.5} 
              dot={false} 
              isAnimationActive={true}
              animationDuration={800}
              strokeDasharray="4 4" 
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <style>{`
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
      `}</style>
    </div>
  );
});
RealTimeChart.displayName = "RealTimeChart";
// ---------------------------------------------

export default function DashboardPage() {
  const {
    predictions, isLoading, isRefreshing,
    lastUpdated, error, refresh,
    bullCount, bearCount, holdCount,
  } = usePredictions();

  const { isOnWatchlist, addTicker, removeTicker } = useWatchlist();
  const [filter, setFilter]           = useState<FilterSignal>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const handleToggleWatchlist = (ticker: string) => {
    if (isOnWatchlist(ticker)) void removeTicker(ticker);
    else void addTicker(ticker);
  };

  const filtered = predictions.filter((p) => {
    const sig = p.data?.finalSignal?.toUpperCase() ?? "";
    const matchFilter =
      filter === "ALL" ||
      (filter === "BUY"  && sig.includes("BUY"))  ||
      (filter === "SELL" && sig.includes("SELL")) ||
      (filter === "HOLD" && sig.includes("HOLD"));
    const matchSearch = !searchQuery || p.ticker.toLowerCase().includes(searchQuery.toLowerCase());
    return matchFilter && matchSearch;
  });

  const totalWithData  = predictions.filter((p) => p.data !== null).length;
  const statValues     = [totalWithData, bullCount, bearCount, holdCount];

  return (
    <div style={{ fontFamily: "var(--font-mono, monospace)" }}>
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 12,
        justifyContent: "space-between", alignItems: "flex-end",
        marginBottom: 24,
        paddingBottom: 18,
        borderBottom: "1px solid rgba(255,255,255,0.05)",
      }}>
        <div>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 6 }}>
            AI SIGNAL DASHBOARD
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.02em", margin: 0 }}>
            Market<span style={{ color: "#06b6d4" }}>_</span>Overview
          </h1>
          <p style={{ fontSize: 11, color: "#334155", marginTop: 4 }}>
            XGBoost + LSTM ensemble · 12 NSE large-caps
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {lastUpdated && (
            <span style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.1em" }}>
              LAST SYNC {lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
          <button
            onClick={refresh}
            disabled={isLoading || isRefreshing}
            style={{
              background: "none",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 3, padding: "6px 14px",
              fontSize: 10, color: isRefreshing ? "#06b6d4" : "#475569",
              cursor: "pointer", letterSpacing: "0.1em",
              display: "flex", alignItems: "center", gap: 7,
              transition: "border-color 0.15s, color 0.15s",
            }}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
              style={{ animation: isRefreshing ? "spin 0.8s linear infinite" : "none" }}>
              <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
              <path d="M21 3v5h-5"/>
              <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
              <path d="M8 16H3v5"/>
            </svg>
            {isRefreshing ? "SYNCING" : "REFRESH"}
          </button>
        </div>
      </div>

      {!isLoading && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8, marginBottom: 24 }}>
          {STAT_CONFIGS.map(({ label, color, sub }, i) => (
            <div key={label} style={{
              background: "#0a0f1a",
              border: `1px solid ${color}22`,
              borderTop: `2px solid ${color}`,
              borderRadius: 4, padding: "14px 16px",
              animation: "fadeUp 0.4s ease both",
              animationDelay: `${i * 60}ms`,
            }}>
              <div style={{ fontSize: 8, color: "#334155", letterSpacing: "0.2em", marginBottom: 8 }}>{label}</div>
              <div style={{ fontSize: 28, fontWeight: 700, color, lineHeight: 1, marginBottom: 4 }}>
                {statValues[i]}
              </div>
              <div style={{ fontSize: 9, color: "#1e293b" }}>{sub}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{
        display: "flex", flexWrap: "wrap", gap: 10,
        justifyContent: "space-between", alignItems: "center",
        marginBottom: 20,
      }}>
        <div style={{
          display: "flex", gap: 2,
          background: "#0a0f1a",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 4, padding: 3,
        }}>
          {(["ALL","BUY","SELL","HOLD"] as FilterSignal[]).map((f) => (
            <button key={f} onClick={() => setFilter(f)}
              style={{
                background: filter === f ? "rgba(6,182,212,0.1)" : "none",
                border: filter === f ? "1px solid rgba(6,182,212,0.25)" : "1px solid transparent",
                borderRadius: 3, padding: "5px 14px",
                fontSize: 10, letterSpacing: "0.1em",
                color: filter === f ? "#06b6d4" : "#334155",
                cursor: "pointer", fontFamily: "inherit",
                transition: "all 0.15s",
              }}>
              {f === "ALL" ? `ALL (${predictions.length})` : f}
            </button>
          ))}
        </div>

        <div style={{ position: "relative" }}>
          <svg style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "#334155" }}
            width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
          <input
            type="text" placeholder="SEARCH TICKER..." value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              background: "#0a0f1a",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 3, padding: "7px 12px 7px 30px",
              fontSize: 10, color: "#94a3b8",
              fontFamily: "inherit", letterSpacing: "0.1em",
              outline: "none", width: 180,
            }}
          />
        </div>
      </div>

      {error && (
        <div style={{
          marginBottom: 20, padding: "10px 14px",
          background: "rgba(244,63,94,0.07)",
          borderLeft: "3px solid #f43f5e",
          borderRadius: "0 4px 4px 0",
          fontSize: 11, color: "#f87171",
        }}>
          {error}
        </div>
      )}

      {isLoading ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} style={{
              background: "#0a0f1a", borderRadius: 4,
              border: "1px solid rgba(255,255,255,0.05)",
              padding: "14px 16px 14px 20px",
              borderLeft: "3px solid rgba(255,255,255,0.05)",
            }}>
              <div className="skeleton" style={{ height: 12, width: "45%", borderRadius: 2, marginBottom: 6 }}/>
              <div className="skeleton" style={{ height: 9, width: "65%", borderRadius: 2, marginBottom: 16 }}/>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <div className="skeleton" style={{ height: 42, borderRadius: 3 }}/>
                <div className="skeleton" style={{ height: 42, borderRadius: 3 }}/>
              </div>
              <div className="skeleton" style={{ height: 4, borderRadius: 2, marginBottom: 12 }}/>
              <div className="skeleton" style={{ height: 32, borderRadius: 3 }}/>
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", padding: "80px 0", textAlign: "center",
        }}>
          <div style={{ fontSize: 11, color: "#334155", letterSpacing: "0.2em", marginBottom: 12 }}>
            NO SIGNALS MATCH
          </div>
          <button onClick={() => { setFilter("ALL"); setSearchQuery(""); }}
            style={{
              background: "none", border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 3, padding: "6px 16px",
              fontSize: 10, color: "#475569", cursor: "pointer",
              fontFamily: "inherit", letterSpacing: "0.1em",
            }}>
            CLEAR FILTERS
          </button>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 15 }}>
          {filtered.map((p, i) => (
            <div key={p.ticker} style={{ display: "flex", flexDirection: "column" }}>
              <StockCard 
                ticker={p.ticker} 
                prediction={p.data} 
                error={p.error}
                index={i} 
                onWatchlist={isOnWatchlist(p.ticker)} 
                onToggleWatchlist={handleToggleWatchlist}
              />
              <RealTimeChart ticker={p.ticker} index={i} />
            </div>
          ))}
        </div>
      )}

      {!isLoading && predictions.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <ConfidenceBar predictions={predictions} showSignal animate />
        </div>
      )}

      <div style={{
        marginTop: 20, padding: "12px 16px",
        background: "#0a0f1a",
        border: "1px solid rgba(255,255,255,0.05)",
        borderLeft: "2px solid #334155",
        borderRadius: "0 4px 4px 0",
        fontSize: 9, color: "#1e293b", letterSpacing: "0.08em",
        lineHeight: 1.7,
        textAlign: "center",
      }}>
        FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE. PAST MODEL PERFORMANCE DOES NOT GUARANTEE FUTURE RESULTS.
      </div>

      <style>{`
        @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:none} }
        @keyframes spin { to{transform:rotate(360deg)} }
        @media(max-width:900px){
          div[style*="repeat(3,1fr)"]{grid-template-columns:repeat(2,1fr) !important}
          div[style*="repeat(4,1fr)"]{grid-template-columns:repeat(2,1fr) !important}
        }
        @media(max-width:560px){
          div[style*="repeat(3,1fr)"]{grid-template-columns:1fr !important}
          div[style*="repeat(4,1fr)"]{grid-template-columns:repeat(2,1fr) !important}
        }
      `}</style>
    </div>
  );
}