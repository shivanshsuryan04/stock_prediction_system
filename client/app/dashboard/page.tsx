"use client";

import { useState } from "react";
import { StockCard } from "@/components/ui/StockCard";
import { ConfidenceBar } from "@/components/charts/ConfidenceBar";
import { usePredictions } from "@/hooks/usePredictions";
import { useWatchlist } from "@/hooks/useWatchlist";

type FilterSignal = "ALL" | "BUY" | "SELL" | "HOLD";

const STAT_CONFIGS = [
  { key: "tracked",  label: "TRACKED",  color: "#06b6d4",  sub: "NSE stocks" },
  { key: "bullish",  label: "BULLISH",  color: "#10b981",  sub: "Buy signals" },
  { key: "bearish",  label: "BEARISH",  color: "#f43f5e",  sub: "Sell signals" },
  { key: "neutral",  label: "NEUTRAL",  color: "#f59e0b",  sub: "Hold signals" },
];

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

      {/* ── page header ── */}
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

      {/* ── stat cards ── */}
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

      {/* ── filters + search ── */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 10,
        justifyContent: "space-between", alignItems: "center",
        marginBottom: 20,
      }}>
        {/* filter tabs */}
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

        {/* search */}
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

      {/* ── error ── */}
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

      {/* ── grid ── */}
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
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10 }}>
          {filtered.map((p, i) => (
            <StockCard key={p.ticker} ticker={p.ticker} prediction={p.data} error={p.error}
              index={i} onWatchlist={isOnWatchlist(p.ticker)} onToggleWatchlist={handleToggleWatchlist}/>
          ))}
        </div>
      )}

      {/* ── confidence chart ── */}
      {!isLoading && predictions.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <ConfidenceBar predictions={predictions} showSignal animate />
        </div>
      )}

      {/* ── disclaimer ── */}
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