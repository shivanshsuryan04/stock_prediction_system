"use client";

import { useRouter } from "next/navigation";
import { useAuthStore } from "@/lib/auth.store";
import { useMarketIndices } from "@/hooks/useMarketIndices";

export function Navbar() {
  const { user, logout } = useAuthStore();
  const router = useRouter();
  // 1. Destructure bankNifty from the hook
  const { nifty, sensex, bankNifty } = useMarketIndices();

  const handleLogout = async () => {
    await logout();
    router.replace("/auth/login");
  };

  const initials =
    user?.name.split(" ").map((n) => n[0]).join("").toUpperCase().slice(0, 2) ?? "??";

  function fmt(n: number | null) {
    if (n === null) return "—";
    return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  }
  function fmtPct(n: number | null) {
    if (n === null) return "";
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
  }

  return (
    <header style={{
      position: "sticky", top: 0, zIndex: 50,
      background: "rgba(5,8,16,0.92)",
      backdropFilter: "blur(20px)",
      borderBottom: "1px solid rgba(255,255,255,0.05)",
      fontFamily: "var(--font-mono, monospace)",
    }}>
      {/* top accent line */}
      <div style={{
        height: 2,
        background: "linear-gradient(90deg, #06b6d4, #10b981 40%, #06b6d4 100%)",
        backgroundSize: "200% 100%",
        animation: "slideGradient 4s linear infinite",
      }} />

      <div style={{
        maxWidth: 1280, margin: "0 auto",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 24px", height: 52,
      }}>
        {/* brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 4, background: "#06b6d4",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <svg width="15" height="15" viewBox="0 0 16 16" fill="none">
              <path d="M2 12L6 6L9 9L12 4L14 6" stroke="#050810" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="14" cy="6" r="1.5" fill="#050810" />
            </svg>
          </div>
          <div>
            <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.06em" }}>
              NEURAL<span style={{ color: "#06b6d4" }}>TRADE</span>
            </span>
            <span style={{
              marginLeft: 8, fontSize: 9, padding: "2px 6px", borderRadius: 2,
              background: "rgba(6,182,212,0.1)", color: "#06b6d4",
              letterSpacing: "0.15em", fontWeight: 600, verticalAlign: "middle",
            }}>AI v2</span>
          </div>
        </div>

        {/* live indices */}
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          {/* NIFTY 50 */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em" }}>NIFTY 50</span>
            {nifty.isLoading ? (
              <span style={{ fontSize: 10, color: "#334155" }}>—</span>
            ) : (
              <>
                <span style={{ fontSize: 10, fontWeight: 700, color: nifty.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmt(nifty.price)}
                </span>
                <span style={{ fontSize: 9, color: nifty.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmtPct(nifty.changePercent)}
                </span>
              </>
            )}
          </div>

          <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.06)" }} />

          {/* SENSEX */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em" }}>SENSEX</span>
            {sensex.isLoading ? (
              <span style={{ fontSize: 10, color: "#334155" }}>—</span>
            ) : (
              <>
                <span style={{ fontSize: 10, fontWeight: 700, color: sensex.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmt(sensex.price)}
                </span>
                <span style={{ fontSize: 9, color: sensex.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmtPct(sensex.changePercent)}
                </span>
              </>
            )}
          </div>

          <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.06)" }} />

          {/* BANK NIFTY */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 9, color: "#334155", letterSpacing: "0.1em" }}>BANK NIFTY</span>
            {bankNifty.isLoading ? (
              <span style={{ fontSize: 10, color: "#334155" }}>—</span>
            ) : (
              <>
                <span style={{ fontSize: 10, fontWeight: 700, color: bankNifty.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmt(bankNifty.price)}
                </span>
                <span style={{ fontSize: 9, color: bankNifty.isUp ? "#10b981" : "#f43f5e" }}>
                  {fmtPct(bankNifty.changePercent)}
                </span>
              </>
            )}
          </div>

          <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.06)" }} />

          {/* live dot */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{
              display: "inline-block", width: 6, height: 6, borderRadius: "50%",
              background: "#10b981", animation: "statusPulse 2s ease-in-out infinite",
            }} />
            <span style={{ fontSize: 10, color: "#334155", letterSpacing: "0.12em" }}>NSE LIVE</span>
          </div>

          <div style={{ width: 1, height: 18, background: "rgba(255,255,255,0.06)" }} />

          {/* user */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 28, height: 28, borderRadius: 4,
              background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.3)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 10, fontWeight: 700, color: "#06b6d4", letterSpacing: "0.05em",
            }}>
              {initials}
            </div>
            <div style={{ display: "none" }} className="sm-show">
              <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.3 }}>{user?.name}</div>
              <div style={{ fontSize: 9, color: "#334155" }}>{user?.email}</div>
            </div>
          </div>

          <button
            onClick={handleLogout}
            style={{
              background: "none", border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: 3, padding: "5px 12px", fontSize: 10, color: "#475569",
              cursor: "pointer", letterSpacing: "0.1em", transition: "border-color 0.15s, color 0.15s",
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(244,63,94,0.4)";
              (e.currentTarget as HTMLButtonElement).style.color = "#f87171";
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(255,255,255,0.07)";
              (e.currentTarget as HTMLButtonElement).style.color = "#475569";
            }}
          >SIGN OUT</button>
        </div>
      </div>

      <style>{`
        @keyframes slideGradient { to { background-position: -200% 0; } }
        @keyframes statusPulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
        @media(min-width:640px){ .sm-show{ display:block !important; } }
      `}</style>
    </header>
  );
}