"use client";

import { useState } from "react";
import { useAuthStore } from "@/lib/auth.store";

interface ToggleProps { label: string; sub: string; value: boolean; onChange: (v: boolean) => void; }
function Toggle({ label, sub, value, onChange }: ToggleProps) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
      <div>
        <div style={{ fontSize: 11, color: "#94a3b8" }}>{label}</div>
        <div style={{ fontSize: 9, color: "#1e293b", marginTop: 2, letterSpacing: "0.08em" }}>{sub}</div>
      </div>
      <button
        onClick={() => onChange(!value)}
        style={{
          width: 30, height: 17, borderRadius: 9,
          background: value ? "#06b6d4" : "#1e293b",
          position: "relative", cursor: "pointer", border: "none",
          transition: "background 0.2s", flexShrink: 0,
        }}
      >
        <span style={{
          width: 11, height: 11, background: "#f1f5f9", borderRadius: "50%",
          position: "absolute", top: 3,
          left: value ? 16 : 3, transition: "left 0.2s",
          display: "block",
        }} />
      </button>
    </div>
  );
}

export default function SettingsPage() {
  const { user } = useAuthStore();

  const initials = user?.name.split(" ").map((n) => n[0]).join("").toUpperCase().slice(0, 2) ?? "??";

  // Model prefs
  const [lstmEnabled,     setLstmEnabled]     = useState(true);
  const [xgbEnabled,      setXgbEnabled]      = useState(true);
  const [ensembleEnabled, setEnsembleEnabled] = useState(true);
  const [autoRefresh,     setAutoRefresh]     = useState(true);
  const [minConf,         setMinConf]         = useState(60);

  // Display prefs
  const [animateCharts,   setAnimateCharts]   = useState(true);
  const [showCacheBadge,  setShowCacheBadge]  = useState(true);
  const [compactMode,     setCompactMode]     = useState(false);

  // Profile form
  const [name,  setName]  = useState(user?.name  ?? "");
  const [email, setEmail] = useState(user?.email ?? "");
  const [saved, setSaved] = useState(false);

  function handleSave() {
    // In a real app you'd call an API here
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  const mono: React.CSSProperties = { fontFamily: "var(--font-mono, monospace)" };
  const card: React.CSSProperties = { background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 4, padding: "16px 18px", marginBottom: 10 };
  const inputStyle: React.CSSProperties = {
    background: "#050810", border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 3, padding: "7px 10px", fontSize: 10, color: "#94a3b8",
    fontFamily: "var(--font-mono, monospace)", outline: "none", width: "100%",
    letterSpacing: "0.06em",
  };
  const sectionLabel: React.CSSProperties = { fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 12 };

  return (
    <div style={mono}>
      {/* Header */}
      <div style={{ marginBottom: 24, paddingBottom: 18, borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 6 }}>CONFIGURATION</div>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", margin: 0 }}>
          Platform<span style={{ color: "#06b6d4" }}>_</span>Settings
        </h1>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Left column */}
        <div>
          {/* Model prefs */}
          <div style={card}>
            <div style={sectionLabel}>MODEL PREFERENCES</div>
            <Toggle label="LSTM Model"      sub="Primary deep learning model"     value={lstmEnabled}     onChange={setLstmEnabled} />
            <Toggle label="XGBoost Model"   sub="Gradient-boosted trees model"    value={xgbEnabled}      onChange={setXgbEnabled} />
            <Toggle label="Ensemble Voting" sub="Combine both model outputs"      value={ensembleEnabled} onChange={setEnsembleEnabled} />
            <Toggle label="Auto Refresh"    sub="Sync predictions every 5 min"   value={autoRefresh}     onChange={setAutoRefresh} />
          </div>

          {/* Confidence threshold */}
          <div style={card}>
            <div style={sectionLabel}>CONFIDENCE THRESHOLD</div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <div style={{ fontSize: 10, color: "#94a3b8" }}>Minimum signal confidence</div>
              <div style={{ fontSize: 10, fontWeight: 700, color: "#06b6d4" }}>{minConf}%</div>
            </div>
            <input
              type="range" min="0" max="100" step="5" value={minConf}
              onChange={(e) => setMinConf(+e.target.value)}
              style={{ width: "100%", accentColor: "#06b6d4", marginBottom: 8 }}
            />
            <div style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.08em", lineHeight: 1.6 }}>
              Signals below this threshold are dimmed in the dashboard and screener.
            </div>
          </div>

          {/* Display */}
          <div style={card}>
            <div style={sectionLabel}>DISPLAY PREFERENCES</div>
            <Toggle label="Animate Charts"    sub="Smooth transitions on load"       value={animateCharts}  onChange={setAnimateCharts} />
            <Toggle label="Show Cached Badge" sub="Indicate when data is from cache" value={showCacheBadge} onChange={setShowCacheBadge} />
            <Toggle label="Compact Mode"      sub="Reduce card padding and spacing"  value={compactMode}    onChange={setCompactMode} />
          </div>
        </div>

        {/* Right column */}
        <div>
          {/* Account */}
          <div style={card}>
            <div style={sectionLabel}>ACCOUNT</div>

            {/* Avatar + info */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, padding: "12px 14px", background: "rgba(6,182,212,0.04)", borderRadius: 3, border: "1px solid rgba(6,182,212,0.1)" }}>
              <div style={{
                width: 42, height: 42, borderRadius: 4,
                background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.3)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 14, fontWeight: 700, color: "#06b6d4",
              }}>
                {initials}
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#f1f5f9", fontWeight: 700 }}>{user?.name ?? "—"}</div>
                <div style={{ fontSize: 10, color: "#334155", marginTop: 2 }}>{user?.email ?? "—"}</div>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.15em", marginBottom: 5 }}>FULL NAME</div>
                <input
                  style={inputStyle} value={name}
                  onChange={(e) => setName(e.target.value)}
                  onFocus={(e) => { (e.target as HTMLInputElement).style.borderColor = "rgba(6,182,212,0.4)"; (e.target as HTMLInputElement).style.color = "#f1f5f9"; }}
                  onBlur={(e)  => { (e.target as HTMLInputElement).style.borderColor = "rgba(255,255,255,0.08)"; (e.target as HTMLInputElement).style.color = "#94a3b8"; }}
                />
              </div>
              <div>
                <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.15em", marginBottom: 5 }}>EMAIL</div>
                <input
                  style={inputStyle} value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={(e) => { (e.target as HTMLInputElement).style.borderColor = "rgba(6,182,212,0.4)"; (e.target as HTMLInputElement).style.color = "#f1f5f9"; }}
                  onBlur={(e)  => { (e.target as HTMLInputElement).style.borderColor = "rgba(255,255,255,0.08)"; (e.target as HTMLInputElement).style.color = "#94a3b8"; }}
                />
              </div>
            </div>

            <button
              onClick={handleSave}
              style={{
                marginTop: 14,
                background: saved ? "rgba(16,185,129,0.1)" : "rgba(6,182,212,0.1)",
                border: `1px solid ${saved ? "rgba(16,185,129,0.3)" : "rgba(6,182,212,0.3)"}`,
                borderRadius: 3, padding: "7px 16px",
                fontSize: 10, letterSpacing: "0.1em",
                color: saved ? "#10b981" : "#06b6d4",
                cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
              }}
            >
              {saved ? "✓ SAVED" : "SAVE CHANGES"}
            </button>
          </div>

          {/* About */}
          <div style={card}>
            <div style={sectionLabel}>ABOUT</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                { label: "Platform",  val: "NeuralTrade AI v2" },
                { label: "Models",    val: "LSTM + XGBoost Ensemble" },
                { label: "Exchange",  val: "NSE (National Stock Exchange)" },
                { label: "Tracked",   val: "12 NSE Large-Cap Stocks" },
                { label: "Refresh",   val: "Every 5 minutes" },
              ].map(({ label, val }) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "5px 0", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                  <span style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em" }}>{label.toUpperCase()}</span>
                  <span style={{ fontSize: 10, color: "#94a3b8" }}>{val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Danger zone */}
          <div style={{ ...card, border: "1px solid rgba(244,63,94,0.15)" }}>
            <div style={{ ...sectionLabel, color: "#f43f5e" }}>DANGER ZONE</div>
            <div style={{ fontSize: 10, color: "#475569", marginBottom: 12, lineHeight: 1.5 }}>
              These actions are irreversible. Proceed with caution.
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button style={{ background: "none", border: "1px solid rgba(244,63,94,0.2)", borderRadius: 3, padding: "6px 12px", fontSize: 10, color: "#f43f5e", cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.08em" }}>
                CLEAR WATCHLIST
              </button>
              <button style={{ background: "none", border: "1px solid rgba(244,63,94,0.2)", borderRadius: 3, padding: "6px 12px", fontSize: 10, color: "#f43f5e", cursor: "pointer", fontFamily: "inherit", letterSpacing: "0.08em" }}>
                RESET PREFERENCES
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 8, padding: "10px 14px", background: "#0a0f1a", border: "1px solid rgba(255,255,255,0.05)", borderLeft: "2px solid #334155", borderRadius: "0 4px 4px 0", fontSize: 9, color: "#1e293b", letterSpacing: "0.08em", textAlign: "center",}}>
        FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
      </div>
    </div>
  );
}