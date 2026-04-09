"use client";

import { useState, useMemo } from "react";
import { usePredictions } from "@/hooks/usePredictions";
import { useWatchlist } from "@/hooks/useWatchlist";
import { SignalBadge } from "@/components/ui/SignalBadge";
import { getSignalMeta } from "@/types";

interface AlertRule {
  id: string;
  ticker: string;
  condition: string;
  notifyInApp: boolean;
  notifyEmail: boolean;
  createdAt: string;
  triggered: boolean;
}

const SUPPORTED_TICKERS = [
  "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
  "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "WIPRO.NS",
  "HCLTECH.NS", "ITC.NS", "MARUTI.NS", "BHARTIARTL.NS",
];

const CONDITIONS = [
  "Signal changes to Strong Buy",
  "Signal changes to Buy",
  "Signal changes to Hold",
  "Signal changes to Sell",
  "Confidence above 80%",
  "Confidence above 90%",
  "Signal reversal (any)",
];

// Static market news — in a real app these would come from a news API
const NEWS = [
  { tag: "MACRO",    tagColor: "#06b6d4", title: "RBI policy meeting minutes signal cautious stance on rate cuts amid inflation", time: "Today, 10:30" },
  { tag: "EARNINGS", tagColor: "#10b981", title: "IT sector Q4 results season kicks off with mixed signals from TCS and Infosys", time: "Today, 09:15" },
  { tag: "SECTOR",   tagColor: "#8b5cf6", title: "Banking NII growth slows; analysts revise HDFC Bank and ICICI Bank targets", time: "Yesterday" },
  { tag: "SECTOR",   tagColor: "#f59e0b", title: "Auto sector facing headwinds from rising input costs and EV transition uncertainty", time: "Yesterday" },
  { tag: "MACRO",    tagColor: "#06b6d4", title: "FII outflows continue for third consecutive week; DII buying provides support", time: "2 days ago" },
  { tag: "ENERGY",   tagColor: "#f43f5e", title: "Reliance Industries announces major capex expansion in green energy segment", time: "2 days ago" },
];

export default function AlertsPage() {
  const { predictions, isLoading } = usePredictions();
  const { items: watchlistItems } = useWatchlist();

  const [alerts, setAlerts] = useState<AlertRule[]>([]);
  const [newTicker, setNewTicker] = useState(SUPPORTED_TICKERS[0]);
  const [newCondition, setNewCondition] = useState(CONDITIONS[0]);
  const [notifyInApp, setNotifyInApp] = useState(true);
  const [notifyEmail, setNotifyEmail] = useState(false);
  const [justCreated, setJustCreated] = useState(false);

  // Auto-generate signal-change alerts from real prediction data
  const liveAlerts = useMemo(() => {
    return predictions
      .filter((p) => p.data !== null)
      .map((p) => {
        const meta = getSignalMeta(p.data!.finalSignal);
        const conf = (p.data!.lstmConf * 100).toFixed(0);
        const time = new Date(p.data!.cachedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return {
          ticker:  p.ticker.replace(".NS", ""),
          signal:  p.data!.finalSignal,
          conf,
          time,
          color:   meta.dotColor,
          bg:      meta.bgStyle,
          isBull:  meta.direction === "bull",
          isBear:  meta.direction === "bear",
        };
      })
      .sort((a, b) => parseFloat(b.conf) - parseFloat(a.conf))
      .slice(0, 6);
  }, [predictions]);

  // High-confidence signals (≥ 80%) from watchlist
  const highConfWatchlist = useMemo(() => {
    return watchlistItems
      .filter((i) => i.prediction && i.prediction.lstmConf >= 0.8)
      .slice(0, 3);
  }, [watchlistItems]);

  function createAlert() {
    const rule: AlertRule = {
      id:          `alert-${Date.now()}`,
      ticker:      newTicker,
      condition:   newCondition,
      notifyInApp,
      notifyEmail,
      createdAt:   new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      triggered:   false,
    };
    setAlerts((prev) => [rule, ...prev]);
    setJustCreated(true);
    setTimeout(() => setJustCreated(false), 2000);
  }

  function deleteAlert(id: string) {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }

  const mono: React.CSSProperties = { fontFamily: "var(--font-mono, monospace)" };

  return (
    <div style={mono}>
      {/* Header */}
      <div style={{ marginBottom: 24, paddingBottom: 18, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 6 }}>SIGNAL MONITORING</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", margin: 0 }}>
          Alerts<span style={{ color: "#06b6d4" }}>_</span>Feed
        </h1>
        <p style={{ fontSize: 11, color: "#334155", marginTop: 4 }}>
          Live signal changes · {liveAlerts.length} active signals
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {/* Live signal alerts from real data */}
        <div style={{ background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 4, padding: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em" }}>LIVE SIGNALS</div>
            {!isLoading && (
              <span style={{ fontSize: 9, background: "rgba(6,182,212,0.1)", color: "#06b6d4", padding: "2px 7px", borderRadius: 2 }}>
                {liveAlerts.length} ACTIVE
              </span>
            )}
          </div>

          {isLoading && (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {[1, 2, 3].map((i) => (
                <div key={i} style={{ height: 50, background: "rgba(255,255,255,0.02)", borderRadius: 3, animation: "pulse 1.4s ease-in-out infinite", animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          )}

          {!isLoading && liveAlerts.map((a, i) => (
            <div key={a.ticker} style={{
              display: "flex", alignItems: "flex-start", gap: 10, padding: "9px 10px",
              borderRadius: 3, marginBottom: 6,
              background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)",
              borderLeft: `3px solid ${a.color}`,
              animation: "fadeUp 0.3s ease both", animationDelay: `${i * 40}ms`,
            }}>
              <div style={{ width: 28, height: 28, borderRadius: 3, background: a.bg, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                <span style={{ fontSize: 10, color: a.color, fontWeight: 700 }}>
                  {a.isBull ? "▲" : a.isBear ? "▼" : "◆"}
                </span>
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.04em" }}>{a.ticker}</span>
                  <SignalBadge signal={a.signal} size="sm" />
                </div>
                <div style={{ fontSize: 9, color: "#334155" }}>LSTM confidence: {a.conf}%</div>
              </div>
              <div style={{ fontSize: 9, color: "#1e293b", flexShrink: 0 }}>{a.time}</div>
            </div>
          ))}
        </div>

        {/* Create alert form */}
        <div style={{ background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 4, padding: 14 }}>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 12 }}>CREATE ALERT</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div>
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.15em", marginBottom: 5 }}>TICKER</div>
              <select
                value={newTicker} onChange={(e) => setNewTicker(e.target.value)}
                style={{ background: "#050810", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 3, padding: "7px 10px", fontSize: 10, color: "#94a3b8", fontFamily: "inherit", outline: "none", width: "100%", letterSpacing: "0.06em" }}
              >
                {SUPPORTED_TICKERS.map((t) => (
                  <option key={t} value={t}>{t.replace(".NS", "")}</option>
                ))}
              </select>
            </div>
            <div>
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.15em", marginBottom: 5 }}>CONDITION</div>
              <select
                value={newCondition} onChange={(e) => setNewCondition(e.target.value)}
                style={{ background: "#050810", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 3, padding: "7px 10px", fontSize: 10, color: "#94a3b8", fontFamily: "inherit", outline: "none", width: "100%", letterSpacing: "0.06em" }}
              >
                {CONDITIONS.map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.15em", marginBottom: 6 }}>NOTIFY VIA</div>
              <div style={{ display: "flex", gap: 16 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "#475569", cursor: "pointer" }}>
                  <input type="checkbox" checked={notifyInApp} onChange={(e) => setNotifyInApp(e.target.checked)} style={{ accentColor: "#06b6d4" }} />
                  IN-APP
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10, color: "#475569", cursor: "pointer" }}>
                  <input type="checkbox" checked={notifyEmail} onChange={(e) => setNotifyEmail(e.target.checked)} style={{ accentColor: "#06b6d4" }} />
                  EMAIL
                </label>
              </div>
            </div>
            <button
              onClick={createAlert}
              style={{
                background: justCreated ? "rgba(16,185,129,0.1)" : "rgba(6,182,212,0.1)",
                border: `1px solid ${justCreated ? "rgba(16,185,129,0.3)" : "rgba(6,182,212,0.3)"}`,
                borderRadius: 3, padding: "8px 14px",
                fontSize: 10, letterSpacing: "0.1em",
                color: justCreated ? "#10b981" : "#06b6d4",
                cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
              }}
            >
              {justCreated ? "✓ ALERT CREATED" : "+ CREATE ALERT"}
            </button>
          </div>
        </div>
      </div>

      {/* User-created alerts */}
      {alerts.length > 0 && (
        <div style={{ background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 4, padding: 14, marginBottom: 16 }}>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 12 }}>MY ALERTS</div>
          {alerts.map((a) => (
            <div key={a.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 10px", borderRadius: 3, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)", marginBottom: 6, animation: "fadeUp 0.2s ease" }}>
              <div>
                <span style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.04em" }}>{a.ticker.replace(".NS", "")}</span>
                <span style={{ fontSize: 9, color: "#475569", marginLeft: 10 }}>{a.condition}</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ display: "flex", gap: 4 }}>
                  {a.notifyInApp && <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 2, background: "rgba(6,182,212,0.08)", color: "#06b6d4", letterSpacing: "0.1em" }}>IN-APP</span>}
                  {a.notifyEmail && <span style={{ fontSize: 8, padding: "1px 5px", borderRadius: 2, background: "rgba(139,92,246,0.08)", color: "#8b5cf6", letterSpacing: "0.1em" }}>EMAIL</span>}
                </div>
                <span style={{ fontSize: 9, color: "#1e293b" }}>{a.createdAt}</span>
                <button onClick={() => deleteAlert(a.id)} style={{ background: "none", border: "none", cursor: "pointer", color: "#1e293b", fontSize: 12, padding: "2px 6px", borderRadius: 2, transition: "color .15s" }}
                  onMouseEnter={e => (e.currentTarget.style.color = "#f87171")}
                  onMouseLeave={e => (e.currentTarget.style.color = "#1e293b")}
                >✕</button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Watchlist high-conf */}
      {highConfWatchlist.length > 0 && (
        <div style={{ background: "#0a0f1a", border: "1px solid rgba(16,185,129,0.15)", borderRadius: 4, padding: 14, marginBottom: 16 }}>
          <div style={{ fontSize: 9, color: "#10b981", letterSpacing: "0.2em", marginBottom: 12 }}>
            ★ WATCHLIST · HIGH CONFIDENCE (≥80%)
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
            {highConfWatchlist.map((item) => {
              const meta = getSignalMeta(item.prediction!.finalSignal);
              return (
                <div key={item.id} style={{ padding: "10px 12px", borderRadius: 3, background: meta.bgStyle, border: `1px solid ${meta.dotColor}33` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 700, color: "#f1f5f9" }}>{item.ticker.replace(".NS", "")}</span>
                    <span style={{ fontSize: 10, fontWeight: 700, color: meta.dotColor }}>{(item.prediction!.lstmConf * 100).toFixed(0)}%</span>
                  </div>
                  <SignalBadge signal={item.prediction!.finalSignal} size="sm" />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Market news */}
      <div style={{ background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 4, padding: 14 }}>
        <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 12 }}>MARKET NEWS</div>
        {NEWS.map((n, i) => (
          <div key={i} style={{ padding: "10px 0", borderBottom: i < NEWS.length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none", cursor: "pointer", transition: "padding-left .15s" }}
            onMouseEnter={e => (e.currentTarget.style.paddingLeft = "6px")}
            onMouseLeave={e => (e.currentTarget.style.paddingLeft = "0")}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{ fontSize: 8, color: n.tagColor, letterSpacing: "0.15em", fontWeight: 700, background: `${n.tagColor}18`, padding: "1px 6px", borderRadius: 2 }}>{n.tag}</span>
              <span style={{ fontSize: 9, color: "#1e293b" }}>{n.time}</span>
            </div>
            <div style={{ fontSize: 10, color: "#94a3b8", lineHeight: 1.4 }}>{n.title}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 20, padding: "10px 14px", background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.05)", borderLeft: "2px solid #334155", borderRadius: "0 4px 4px 0", fontSize: 9, color: "#1e293b", letterSpacing: "0.08em",textAlign: "center",}}>
        AI SIGNALS ARE FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
      </div>

      <style>{`
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
        @keyframes pulse  { 0%,100%{opacity:.3} 50%{opacity:.6} }
      `}</style>
    </div>
  );
}