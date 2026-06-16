"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/lib/auth.store";

// ── Fake ticker data for the animated left panel ──────────────────────────────
const TICKERS = [
  { sym: "RELIANCE", price: 2847.35, change: +1.24 },
  { sym: "TCS", price: 3921.1, change: -0.43 },
  { sym: "INFY", price: 1876.55, change: +2.11 },
  { sym: "HDFCBANK", price: 1654.2, change: -0.78 },
  { sym: "ICICIBANK", price: 1123.45, change: +0.92 },
  { sym: "SBIN", price: 812.3, change: +1.67 },
  { sym: "AXISBANK", price: 987.65, change: -1.34 },
  { sym: "WIPRO", price: 524.8, change: +0.55 },
  { sym: "HCLTECH", price: 1432.9, change: +3.21 },
  { sym: "ITC", price: 498.15, change: -0.22 },
  { sym: "MARUTI", price: 12340.0, change: +0.88 },
  { sym: "BHARTIARTL", price: 1289.75, change: +1.45 },
];

function useLiveTickerData() {
  const [data, setData] = useState(TICKERS);
  useEffect(() => {
    const id = setInterval(() => {
      setData((prev) =>
        prev.map((t) => ({
          ...t,
          price: parseFloat(
            (t.price * (1 + (Math.random() - 0.499) * 0.002)).toFixed(2),
          ),
          change: parseFloat(
            (t.change + (Math.random() - 0.5) * 0.1).toFixed(2),
          ),
        })),
      );
    }, 1200);
    return () => clearInterval(id);
  }, []);
  return data;
}

// ── Mini sparkline component ──────────────────────────────────────────────────
function Sparkline({ positive }: { positive: boolean }) {
  const points = useRef(
    Array.from(
      { length: 20 },
      (_, i) => 50 + Math.sin(i * 0.6) * 20 + Math.random() * 15,
    ),
  );
  const pts = points.current.map((y, x) => `${x * 5},${100 - y}`).join(" ");
  const color = positive ? "#10b981" : "#f43f5e";
  return (
    <svg width="60" height="24" viewBox="0 0 95 100" preserveAspectRatio="none">
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="2.5"
        strokeLinejoin="round"
        opacity="0.8"
      />
    </svg>
  );
}

export default function LoginPage() {
  const router = useRouter();
  const { login, isAuthenticated, isLoading } = useAuthStore();
  const tickers = useLiveTickerData();

  const [form, setForm] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [apiError, setApiError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [focusedField, setFocusedField] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoading && isAuthenticated) router.replace("/dashboard");
  }, [isAuthenticated, isLoading, router]);

  const validate = () => {
    const e: Record<string, string> = {};
    if (!form.email) e.email = "Required";
    else if (!/\S+@\S+\.\S+/.test(form.email)) e.email = "Invalid email";
    if (!form.password) e.password = "Required";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    setApiError("");
    try {
      await login(form.email, form.password);
      router.replace("/dashboard");
    } catch (err: unknown) {
      setApiError(
        (err as { response?: { data?: { message?: string } } })?.response?.data
          ?.message ?? "Invalid credentials.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  if (isLoading) return null;

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        fontFamily: "'DM Mono', monospace",
        background: "#050810",
      }}
    >
      {/* ── LEFT PANEL — live market data ── */}
      <div
        style={{
          display: "none",
          flex: "0 0 52%",
          padding: "0",
          overflowY: "hidden",
          position: "relative",
          borderRight: "1px solid rgba(255,255,255,0.06)",
        }}
        className="auth-left-panel"
      >
        {/* scanline overlay */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            zIndex: 2,
            backgroundImage:
              "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)",
          }}
        />

        {/* top bar */}
        <div
          style={{
            padding: "20px 32px 16px",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#10b981",
                boxShadow: "0 0 8px #10b981",
                animation: "pulse 2s ease-in-out infinite",
              }}
            />
            <span
              style={{
                fontSize: 11,
                color: "#4ade80",
                letterSpacing: "0.12em",
              }}
            >
              NSE LIVE
            </span>
          </div>
          <span
            style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em" }}
          >
            {new Date().toLocaleTimeString("en-IN", { hour12: false })} IST
          </span>
        </div>

        {/* heading */}
        <div style={{ padding: "40px 32px 24px" }}>
          <div
            style={{
              fontSize: 10,
              color: "#334155",
              letterSpacing: "0.2em",
              marginBottom: 12,
            }}
          >
            AI-POWERED SIGNALS
          </div>
          <div
            style={{
              fontSize: 38,
              fontWeight: 300,
              lineHeight: 1.15,
              color: "#f1f5f9",
              letterSpacing: "-0.02em",
              fontFamily: "'DM Serif Display', serif",
            }}
          >
            Predict the
            <br />
            <span style={{ color: "#06b6d4" }}>market</span>
            <br />
            before it moves.
          </div>
          <div
            style={{
              marginTop: 16,
              fontSize: 12,
              color: "#475569",
              lineHeight: 1.6,
            }}
          >
            XGBoost + LSTM ensemble
            <br />
            trained on NSE historical data.
          </div>
        </div>

        {/* ticker table */}
        <div style={{ padding: "0 32px" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto auto auto",
              gap: "0 16px",
              fontSize: 10,
              color: "#334155",
              letterSpacing: "0.1em",
              paddingBottom: 8,
              borderBottom: "1px solid rgba(255,255,255,0.05)",
            }}
          >
            <span>SYMBOL</span>
            <span>PRICE</span>
            <span>CHG%</span>
            <span>7D</span>
          </div>
          {tickers.map((t) => {
            const pos = t.change >= 0;
            return (
              <div
                key={t.sym}
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr auto auto auto",
                  gap: "0 16px",
                  alignItems: "center",
                  padding: "9px 0",
                  borderBottom: "1px solid rgba(255,255,255,0.03)",
                  transition: "background 0.3s",
                }}
              >
                <span
                  style={{ fontSize: 11, color: "#94a3b8", fontWeight: 500 }}
                >
                  {t.sym}
                </span>
                <span
                  style={{
                    fontSize: 12,
                    color: "#e2e8f0",
                    fontVariantNumeric: "tabular-nums",
                    minWidth: 72,
                    textAlign: "right",
                  }}
                >
                  ₹
                  {t.price.toLocaleString("en-IN", {
                    minimumFractionDigits: 2,
                  })}
                </span>
                <span
                  style={{
                    fontSize: 11,
                    minWidth: 52,
                    textAlign: "right",
                    color: pos ? "#10b981" : "#f43f5e",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {pos ? "+" : ""}
                  {t.change.toFixed(2)}%
                </span>
                <Sparkline positive={pos} />
              </div>
            );
          })}
        </div>

        {/* bottom model badges */}
        <div style={{ padding: "24px 32px", display: "flex", gap: 8 }}>
          {["XGBoost", "LSTM", "Ensemble"].map((m) => (
            <div
              key={m}
              style={{
                fontSize: 10,
                padding: "4px 10px",
                borderRadius: 4,
                background: "rgba(6,182,212,0.08)",
                border: "1px solid rgba(6,182,212,0.2)",
                color: "#06b6d4",
                letterSpacing: "0.1em",
              }}
            >
              {m}
            </div>
          ))}
        </div>
      </div>

      {/* ── RIGHT PANEL — login form ── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "40px 24px",
          position: "relative",
        }}
      >
        {/* background grid */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            backgroundImage: `
            linear-gradient(rgba(6,182,212,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6,182,212,0.03) 1px, transparent 1px)
          `,
            backgroundSize: "40px 40px",
          }}
        />
        {/* corner glow */}
        <div
          style={{
            position: "absolute",
            bottom: 0,
            right: 0,
            width: 400,
            height: 400,
            background:
              "radial-gradient(circle at bottom right, rgba(6,182,212,0.06), transparent 70%)",
            pointerEvents: "none",
          }}
        />

        <div
          style={{
            width: "100%",
            maxWidth: 380,
            position: "relative",
            zIndex: 1,
          }}
        >
          {/* logo */}
          <div style={{ marginBottom: 48 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 32,
              }}
            >
              <div
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: 8,
                  background: "#06b6d4",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                <svg width="18" height="18" viewBox="0 0 16 16" fill="none">
                  <path
                    d="M2 12L6 6L9 9L12 4L14 6"
                    stroke="#050810"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="14" cy="6" r="1.5" fill="#050810" />
                </svg>
              </div>
              <span
                style={{
                  fontSize: 15,
                  fontWeight: 600,
                  color: "#f1f5f9",
                  letterSpacing: "-0.01em",
                  fontFamily: "var(--font-display, sans-serif)",
                }}
              >
                NeuralTrade
              </span>
            </div>

            <div
              style={{
                fontSize: 26,
                fontWeight: 300,
                color: "#f1f5f9",
                letterSpacing: "-0.02em",
                lineHeight: 1.2,
                fontFamily: "var(--font-display, sans-serif)",
              }}
            >
              Welcome back
            </div>
            <div style={{ marginTop: 6, fontSize: 13, color: "#475569" }}>
              Sign in to access your AI signals
            </div>
          </div>

          {/* form */}
          <form onSubmit={handleSubmit} noValidate>
            {/* api error */}
            {apiError && (
              <div
                style={{
                  marginBottom: 20,
                  padding: "11px 14px",
                  background: "rgba(244,63,94,0.07)",
                  border: "1px solid rgba(244,63,94,0.2)",
                  borderRadius: 8,
                  fontSize: 12,
                  color: "#fca5a5",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  style={{ flexShrink: 0 }}
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {apiError}
              </div>
            )}

            {/* email */}
            <div style={{ marginBottom: 16 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  color: "#475569",
                  marginBottom: 6,
                  letterSpacing: "0.1em",
                }}
              >
                EMAIL
              </label>
              <div style={{ position: "relative" }}>
                <input
                  type="email"
                  autoComplete="email"
                  placeholder="you@example.com"
                  value={form.email}
                  onChange={(e) =>
                    setForm((p) => ({ ...p, email: e.target.value }))
                  }
                  onFocus={() => setFocusedField("email")}
                  onBlur={() => setFocusedField(null)}
                  style={{
                    width: "100%",
                    padding: "12px 14px",
                    fontSize: 13,
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${errors.email ? "rgba(244,63,94,0.5)" : focusedField === "email" ? "#06b6d4" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: 8,
                    color: "#f1f5f9",
                    outline: "none",
                    boxSizing: "border-box",
                    fontFamily: "inherit",
                    transition: "border-color 0.15s",
                    boxShadow:
                      focusedField === "email"
                        ? "0 0 0 3px rgba(6,182,212,0.1)"
                        : "none",
                  }}
                />
              </div>
              {errors.email && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>
                  {errors.email}
                </div>
              )}
            </div>

            {/* password */}
            <div style={{ marginBottom: 28 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: 6,
                }}
              >
                <label
                  style={{
                    fontSize: 11,
                    color: "#475569",
                    letterSpacing: "0.1em",
                  }}
                >
                  PASSWORD
                </label>
              </div>
              <div style={{ position: "relative" }}>
                <input
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  placeholder="••••••••"
                  value={form.password}
                  onChange={(e) =>
                    setForm((p) => ({ ...p, password: e.target.value }))
                  }
                  onFocus={() => setFocusedField("password")}
                  onBlur={() => setFocusedField(null)}
                  style={{
                    width: "100%",
                    padding: "12px 44px 12px 14px",
                    fontSize: 13,
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${errors.password ? "rgba(244,63,94,0.5)" : focusedField === "password" ? "#06b6d4" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: 8,
                    color: "#f1f5f9",
                    outline: "none",
                    boxSizing: "border-box",
                    fontFamily: "inherit",
                    transition: "border-color 0.15s",
                    boxShadow:
                      focusedField === "password"
                        ? "0 0 0 3px rgba(6,182,212,0.1)"
                        : "none",
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((p) => !p)}
                  style={{
                    position: "absolute",
                    right: 12,
                    top: "50%",
                    transform: "translateY(-50%)",
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    color: "#475569",
                    padding: 4,
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  {showPassword ? (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg
                      width="15"
                      height="15"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                    >
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
              {errors.password && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>
                  {errors.password}
                </div>
              )}
            </div>

            {/* submit */}
            <button
              type="submit"
              disabled={submitting}
              style={{
                width: "100%",
                padding: "13px",
                fontSize: 13,
                background: submitting ? "rgba(6,182,212,0.5)" : "#06b6d4",
                border: "none",
                borderRadius: 8,
                color: "#050810",
                fontWeight: 700,
                cursor: submitting ? "not-allowed" : "pointer",
                fontFamily: "inherit",
                letterSpacing: "0.05em",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                transition: "all 0.15s",
              }}
            >
              {submitting ? (
                <>
                  <span
                    style={{
                      width: 14,
                      height: 14,
                      border: "2px solid rgba(5,8,16,0.3)",
                      borderTopColor: "#050810",
                      borderRadius: "50%",
                      animation: "spin 0.7s linear infinite",
                      display: "inline-block",
                    }}
                  />
                  SIGNING IN...
                </>
              ) : (
                "SIGN IN →"
              )}
            </button>
          </form>

          {/* divider */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              margin: "24px 0",
            }}
          >
            <div
              style={{
                flex: 1,
                height: 1,
                background: "rgba(255,255,255,0.06)",
              }}
            />
            <span
              style={{ fontSize: 11, color: "#334155", letterSpacing: "0.1em" }}
            >
              NO ACCOUNT?
            </span>
            <div
              style={{
                flex: 1,
                height: 1,
                background: "rgba(255,255,255,0.06)",
              }}
            />
          </div>

          <Link
            href="/auth/register"
            style={{
              display: "block",
              width: "100%",
              padding: "12px",
              textAlign: "center",
              fontSize: 13,
              color: "#94a3b8",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              textDecoration: "none",
              letterSpacing: "0.05em",
              transition: "all 0.15s",
              background: "rgba(255,255,255,0.02)",
              boxSizing: "border-box",
            }}
          >
            CREATE ACCOUNT
          </Link>
          <Link
            href="/dashboard"
            style={{
              display: "block",
              width: "100%",
              padding: "12px",
              textAlign: "center",
              fontSize: "13px",
              color: "#06b6d4",
              border: "1px solid rgba(6,182,212,0.25)",
              borderRadius: "8px",
              textDecoration: "none",
              letterSpacing: "0.05em",
              marginTop: "12px",
              boxSizing: "border-box",
            }}
          >
            VIEW DEMO
          </Link>

          <div
            style={{
              marginTop: 32,
              fontSize: 10,
              color: "#1e293b",
              textAlign: "center",
              lineHeight: 1.6,
            }}
          >
            FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Serif+Display&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @media (min-width: 900px) {
          .auth-left-panel { display: flex !important; flex-direction: column; }
        }
        input::placeholder { color: #1e293b; }
        input:-webkit-autofill {
          -webkit-box-shadow: 0 0 0 100px #050810 inset !important;
          -webkit-text-fill-color: #f1f5f9 !important;
        }
      `}</style>
    </div>
  );
}
