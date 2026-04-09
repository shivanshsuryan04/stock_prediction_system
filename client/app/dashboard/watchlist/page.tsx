"use client";

import { useState } from "react";
import { SignalBadge } from "@/components/ui/SignalBadge";
import { useWatchlist } from "@/hooks/useWatchlist";
import { getSignalMeta } from "@/types";
import type { WatchlistItem } from "@/types";

const SUPPORTED_TICKERS = [
  "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
  "SBIN.NS","AXISBANK.NS","WIPRO.NS","HCLTECH.NS","ITC.NS",
  "MARUTI.NS","BHARTIARTL.NS",
];

const COMPANY_NAMES: Record<string, string> = {
  "RELIANCE.NS":"Reliance Industries","TCS.NS":"Tata Consultancy",
  "INFY.NS":"Infosys","HDFCBANK.NS":"HDFC Bank",
  "ICICIBANK.NS":"ICICI Bank","SBIN.NS":"State Bank of India",
  "AXISBANK.NS":"Axis Bank","WIPRO.NS":"Wipro",
  "HCLTECH.NS":"HCL Technologies","ITC.NS":"ITC Limited",
  "MARUTI.NS":"Maruti Suzuki","BHARTIARTL.NS":"Bharti Airtel",
};

export default function WatchlistPage() {
  const { items, isLoading, error, addTicker, removeTicker, isOnWatchlist } = useWatchlist();
  const [showAdd, setShowAdd]     = useState(false);
  const [adding, setAdding]       = useState<string | null>(null);
  const [removing, setRemoving]   = useState<string | null>(null);

  const handleAdd = async (ticker: string) => {
    setAdding(ticker);
    await addTicker(ticker);
    setAdding(null);
  };

  const handleRemove = async (ticker: string) => {
    setRemoving(ticker);
    await removeTicker(ticker);
    setRemoving(null);
  };

  const available = SUPPORTED_TICKERS.filter((t) => !isOnWatchlist(t));

  return (
    <div style={{ fontFamily: "var(--font-mono, monospace)" }}>

      {/* ── header ── */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "flex-end",
        marginBottom: 24, paddingBottom: 18,
        borderBottom: "1px solid rgba(255,255,255,0.05)",
      }}>
        <div>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 6 }}>
            PERSONAL PORTFOLIO
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.02em", margin: 0 }}>
            Watch<span style={{ color: "#06b6d4" }}>_</span>List
          </h1>
          <p style={{ fontSize: 11, color: "#334155", marginTop: 4 }}>
            {items.length} stock{items.length !== 1 ? "s" : ""} monitored
          </p>
        </div>

        <button
          onClick={() => setShowAdd((v) => !v)}
          style={{
            display: "flex", alignItems: "center", gap: 7,
            background: showAdd ? "rgba(6,182,212,0.1)" : "none",
            border: `1px solid ${showAdd ? "rgba(6,182,212,0.3)" : "rgba(255,255,255,0.08)"}`,
            borderRadius: 3, padding: "7px 16px",
            fontSize: 10, letterSpacing: "0.1em",
            color: showAdd ? "#06b6d4" : "#475569",
            cursor: "pointer", fontFamily: "inherit",
            transition: "all 0.15s",
          }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            {showAdd
              ? <><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></>
              : <><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></>
            }
          </svg>
          {showAdd ? "CLOSE" : "ADD STOCK"}
        </button>
      </div>

      {/* ── add panel ── */}
      {showAdd && (
        <div style={{
          marginBottom: 20, padding: "16px 20px",
          background: "#0a0f1a",
          border: "1px solid rgba(6,182,212,0.2)",
          borderLeft: "3px solid #06b6d4",
          borderRadius: "0 4px 4px 0",
          animation: "fadeUp 0.2s ease",
        }}>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.18em", marginBottom: 12 }}>
            AVAILABLE TO TRACK
          </div>
          {available.length === 0 ? (
            <div style={{ fontSize: 11, color: "#334155" }}>All supported stocks are already on your watchlist.</div>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {available.map((ticker) => (
                <button key={ticker}
                  onClick={() => void handleAdd(ticker)}
                  disabled={adding === ticker}
                  style={{
                    display: "flex", alignItems: "center", gap: 6,
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.07)",
                    borderRadius: 3, padding: "5px 12px",
                    fontSize: 10, letterSpacing: "0.08em",
                    color: adding === ticker ? "#06b6d4" : "#475569",
                    cursor: "pointer", fontFamily: "inherit",
                    transition: "all 0.15s",
                    opacity: adding === ticker ? 0.6 : 1,
                  }}
                >
                  {adding === ticker ? (
                    <span style={{
                      width: 8, height: 8, borderRadius: "50%",
                      border: "1.5px solid #06b6d4", borderTopColor: "transparent",
                      animation: "spin 0.7s linear infinite", display: "inline-block",
                    }}/>
                  ) : (
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
                    </svg>
                  )}
                  {ticker.replace(".NS", "")}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── error ── */}
      {error && (
        <div style={{
          marginBottom: 16, padding: "10px 14px",
          background: "rgba(244,63,94,0.07)", color: "#f87171",
          borderLeft: "3px solid #f43f5e",
          borderRadius: "0 4px 4px 0", fontSize: 11,
        }}>
          {error}
        </div>
      )}

      {/* ── skeleton ── */}
      {isLoading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {[1,2,3].map((i) => (
            <div key={i} style={{
              background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.05)",
              borderRadius: 4, padding: "14px 20px",
              display: "flex", alignItems: "center", gap: 16,
            }}>
              <div className="skeleton" style={{ height: 12, width: 60, borderRadius: 2 }}/>
              <div className="skeleton" style={{ height: 10, width: 140, borderRadius: 2 }}/>
              <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
                <div className="skeleton" style={{ height: 20, width: 60, borderRadius: 2 }}/>
                <div className="skeleton" style={{ height: 20, width: 60, borderRadius: 2 }}/>
                <div className="skeleton" style={{ height: 20, width: 70, borderRadius: 2 }}/>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── empty ── */}
      {!isLoading && items.length === 0 && (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", padding: "80px 0", textAlign: "center",
          background: "#0a0f1a", border: "1px dashed rgba(255,255,255,0.06)",
          borderRadius: 4,
        }}>
          <div style={{
            width: 40, height: 40, borderRadius: 4, marginBottom: 16,
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.07)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="1.5">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
          </div>
          <div style={{ fontSize: 11, color: "#334155", letterSpacing: "0.15em", marginBottom: 4 }}>
            NO STOCKS TRACKED
          </div>
          <div style={{ fontSize: 10, color: "#1e293b" }}>Click ADD STOCK to begin monitoring</div>
        </div>
      )}

      {/* ── table ── */}
      {!isLoading && items.length > 0 && (
        <div style={{
          background: "#0a0f1a",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 4, overflow: "hidden",
        }}>
          {/* thead */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 100px 120px 110px 44px",
            gap: 0, padding: "8px 20px",
            background: "rgba(255,255,255,0.02)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            fontSize: 8, color: "#1e293b", letterSpacing: "0.18em",
          }}>
            <span>STOCK</span>
            <span style={{ textAlign: "center" }}>XGBOOST</span>
            <span style={{ textAlign: "center" }}>LSTM</span>
            <span style={{ textAlign: "center" }}>ENSEMBLE</span>
            <span/>
          </div>

          {/* rows */}
          {items.map((item: WatchlistItem, idx: number) => {
            const p    = item.prediction;
            const meta = p ? getSignalMeta(p.finalSignal) : null;
            const isRemoving = removing === item.ticker;

            return (
              <div key={item.id}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 100px 120px 110px 44px",
                  alignItems: "center", padding: "12px 20px",
                  borderBottom: idx < items.length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none",
                  borderLeft: `3px solid ${meta ? meta.dotColor + "66" : "transparent"}`,
                  background: isRemoving ? "rgba(244,63,94,0.04)" : "transparent",
                  animation: "fadeUp 0.3s ease both",
                  animationDelay: `${idx * 40}ms`,
                  transition: "background 0.2s",
                  opacity: isRemoving ? 0.5 : 1,
                }}
              >
                {/* stock */}
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    {meta && (
                      <div style={{
                        width: 6, height: 6, borderRadius: "50%",
                        background: meta.dotColor, flexShrink: 0,
                        animation: "badgePulse 2s ease-in-out infinite",
                      }}/>
                    )}
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.04em" }}>
                      {item.ticker.replace(".NS", "")}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: "#334155", marginTop: 2, paddingLeft: 14 }}>
                    {COMPANY_NAMES[item.ticker] ?? item.ticker}
                  </div>
                </div>

                {/* XGBoost */}
                <div style={{ textAlign: "center" }}>
                  {p ? <SignalBadge signal={p.xgbSignal} size="sm"/> : <div className="skeleton" style={{ height: 18, width: 56, borderRadius: 2, margin: "0 auto" }}/>}
                </div>

                {/* LSTM */}
                <div style={{ textAlign: "center" }}>
                  {p ? (
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                      <SignalBadge signal={p.lstmSignal} size="sm"/>
                      <span style={{ fontSize: 9, color: "#334155" }}>
                        {(p.lstmConf * 100).toFixed(0)}% conf
                      </span>
                    </div>
                  ) : <div className="skeleton" style={{ height: 18, width: 56, borderRadius: 2, margin: "0 auto" }}/>}
                </div>

                {/* Ensemble */}
                <div style={{ textAlign: "center" }}>
                  {p ? <SignalBadge signal={p.finalSignal} size="sm"/> : <div className="skeleton" style={{ height: 18, width: 66, borderRadius: 2, margin: "0 auto" }}/>}
                </div>

                {/* remove */}
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <button
                    onClick={() => void handleRemove(item.ticker)}
                    disabled={isRemoving}
                    style={{
                      background: "none", border: "none",
                      cursor: isRemoving ? "not-allowed" : "pointer",
                      color: "#1e293b", padding: 6, borderRadius: 3,
                      transition: "color 0.15s, background 0.15s",
                    }}
                    onMouseEnter={e => {
                      if (!isRemoving) {
                        (e.currentTarget as HTMLButtonElement).style.color = "#f87171";
                        (e.currentTarget as HTMLButtonElement).style.background = "rgba(244,63,94,0.08)";
                      }
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLButtonElement).style.color = "#1e293b";
                      (e.currentTarget as HTMLButtonElement).style.background = "none";
                    }}
                    aria-label={`Remove ${item.ticker.replace(".NS","")}`}
                  >
                    {isRemoving ? (
                      <span style={{
                        display: "inline-block", width: 10, height: 10, borderRadius: "50%",
                        border: "1.5px solid #f43f5e", borderTopColor: "transparent",
                        animation: "spin 0.7s linear infinite",
                      }}/>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6l-1 14H6L5 6"/>
                        <path d="M10 11v6M14 11v6M9 6V4h6v2"/>
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* disclaimer */}
      <div style={{
        marginTop: 20, padding: "10px 14px",
        background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.05)",
        borderLeft: "2px solid #334155", borderRadius: "0 4px 4px 0",
        fontSize: 9, color: "#1e293b", letterSpacing: "0.08em", textAlign: "center",
      }}>
        AI SIGNALS ARE FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
      </div>

      <style>{`
        @keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes badgePulse{0%,100%{opacity:1}50%{opacity:0.3}}
      `}</style>
    </div>
  );
}