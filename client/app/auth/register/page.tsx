"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/lib/auth.store";

// ── Animated model accuracy stats for the left panel ─────────────────────────
const MODEL_STATS = [
  { model: "XGBoost", accuracy: 68.4, trades: 2841, winRate: 71.2 },
  { model: "LSTM",    accuracy: 71.1, trades: 2841, winRate: 73.8 },
  { model: "Ensemble",accuracy: 74.6, trades: 2841, winRate: 76.5 },
];

const FEATURE_LIST = [
  "RSI · MACD · Signal Line",
  "MA 10 / 20 / 50",
  "Lag features + volatility",
  "Dist from 50-day average",
  "12 NSE large-cap stocks",
  "Daily retraining pipeline",
];

function AnimatedStat({ value, suffix = "%" }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = value / 40;
    const id = setInterval(() => {
      start += step;
      if (start >= value) { setDisplay(value); clearInterval(id); }
      else setDisplay(parseFloat(start.toFixed(1)));
    }, 30);
    return () => clearInterval(id);
  }, [value]);
  return <>{display.toFixed(1)}{suffix}</>;
}

function AccuracyBar({ value, color }: { value: number; color: string }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const id = setTimeout(() => setWidth(value), 200);
    return () => clearTimeout(id);
  }, [value]);
  return (
    <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
      <div style={{
        height: "100%", borderRadius: 2, background: color,
        width: `${width}%`, transition: "width 1.2s cubic-bezier(0.22,1,0.36,1)",
      }}/>
    </div>
  );
}

export default function RegisterPage() {
  const router = useRouter();
  const { register, isAuthenticated, isLoading } = useAuthStore();

  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [apiError, setApiError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [showPass, setShowPass] = useState(false);
  const [focusedField, setFocusedField] = useState<string | null>(null);
  const [step, setStep] = useState(0); // 0 = identity, 1 = password

  useEffect(() => {
    if (!isLoading && isAuthenticated) router.replace("/dashboard");
  }, [isAuthenticated, isLoading, router]);

  const strength = (() => {
    const p = form.password;
    if (!p) return { score: 0, label: "", color: "" };
    let s = 0;
    if (p.length >= 8) s++;
    if (/[A-Z]/.test(p)) s++;
    if (/[0-9]/.test(p)) s++;
    if (/[^A-Za-z0-9]/.test(p)) s++;
    if (s <= 1) return { score: s, label: "WEAK", color: "#f43f5e" };
    if (s === 2) return { score: s, label: "FAIR", color: "#f59e0b" };
    if (s === 3) return { score: s, label: "GOOD", color: "#06b6d4" };
    return { score: s, label: "STRONG", color: "#10b981" };
  })();

  const validateStep0 = () => {
    const e: Record<string, string> = {};
    if (!form.name.trim() || form.name.trim().length < 2) e.name = "Min 2 characters";
    if (!form.email) e.email = "Required";
    else if (!/\S+@\S+\.\S+/.test(form.email)) e.email = "Invalid email";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const validateStep1 = () => {
    const e: Record<string, string> = {};
    if (!form.password) e.password = "Required";
    else if (form.password.length < 8) e.password = "Min 8 characters";
    else if (!/[A-Z]/.test(form.password)) e.password = "Need one uppercase letter";
    else if (!/[0-9]/.test(form.password)) e.password = "Need one number";
    if (form.password !== form.confirm) e.confirm = "Passwords don't match";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleNext = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateStep0()) setStep(1);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateStep1()) return;
    setSubmitting(true);
    setApiError("");
    try {
      await register(form.name.trim(), form.email, form.password);
      router.replace("/dashboard");
    } catch (err: unknown) {
      setApiError((err as { response?: { data?: { message?: string } } })?.response?.data?.message ?? "Registration failed.");
      setStep(0);
    } finally {
      setSubmitting(false);
    }
  };

  const inputStyle = (field: string, hasError: boolean) => ({
    width: "100%", padding: "12px 14px", fontSize: 13,
    background: "rgba(255,255,255,0.03)",
    border: `1px solid ${hasError ? "rgba(244,63,94,0.5)" : focusedField === field ? "#06b6d4" : "rgba(255,255,255,0.08)"}`,
    borderRadius: 8, color: "#f1f5f9", outline: "none",
    boxSizing: "border-box" as const, fontFamily: "inherit",
    transition: "border-color 0.15s",
    boxShadow: focusedField === field ? "0 0 0 3px rgba(6,182,212,0.1)" : "none",
  });

  if (isLoading) return null;

  return (
    <div style={{
      display: "flex", minHeight: "100vh", fontFamily: "'DM Mono', monospace",
      background: "#050810",
    }}>
      {/* ── LEFT PANEL — model performance ── */}
      <div style={{ display: "none", flex: "0 0 52%", position: "relative", borderRight: "1px solid rgba(255,255,255,0.06)", overflowY: "auto" }}
        className="auth-left-panel">

        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.02) 2px, rgba(0,0,0,0.02) 4px)",
        }}/>

        <div style={{ padding: "28px 32px 0", position: "relative" }}>
          <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.2em", marginBottom: 16 }}>
            MODEL PERFORMANCE · BACKTESTED
          </div>

          <div style={{ fontSize: 34, fontWeight: 300, lineHeight: 1.2, color: "#f1f5f9", letterSpacing: "-0.02em", fontFamily: "'DM Serif Display', serif", marginBottom: 32 }}>
            Three models.<br/>
            One <span style={{ color: "#06b6d4" }}>ensemble</span><br/>
            signal.
          </div>

          {/* model cards */}
          {MODEL_STATS.map((m, i) => {
            const colors = ["#06b6d4", "#10b981", "#f59e0b"];
            const c = colors[i] ?? "#06b6d4";
            return (
              <div key={m.model} style={{
                padding: "16px 18px", marginBottom: 12,
                background: "rgba(255,255,255,0.02)",
                border: `1px solid ${i === 2 ? "rgba(6,182,212,0.2)" : "rgba(255,255,255,0.05)"}`,
                borderRadius: 10,
                animation: `fadeUp 0.5s ease both`,
                animationDelay: `${i * 0.12}s`,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                  <span style={{ fontSize: 11, color: c, letterSpacing: "0.1em" }}>{m.model.toUpperCase()}</span>
                  <span style={{ fontSize: 18, color: "#f1f5f9", fontVariantNumeric: "tabular-nums" }}>
                    <AnimatedStat value={m.accuracy} />
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <AccuracyBar value={m.accuracy} color={c} />
                </div>
                <div style={{ marginTop: 8, display: "flex", gap: 16 }}>
                  <span style={{ fontSize: 10, color: "#334155" }}>
                    {m.trades.toLocaleString()} signals
                  </span>
                  <span style={{ fontSize: 10, color: "#334155" }}>
                    win rate <span style={{ color: c }}>{m.winRate}%</span>
                  </span>
                </div>
              </div>
            );
          })}

          {/* feature list */}
          <div style={{ marginTop: 24, marginBottom: 32 }}>
            <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.15em", marginBottom: 12 }}>
              FEATURES USED
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {FEATURE_LIST.map(f => (
                <div key={f} style={{
                  fontSize: 10, padding: "4px 10px", borderRadius: 4,
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.07)",
                  color: "#64748b", letterSpacing: "0.05em",
                }}>{f}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── RIGHT PANEL — register form ── */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "40px 24px", position: "relative",
      }}>
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: `
            linear-gradient(rgba(6,182,212,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(6,182,212,0.025) 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px",
        }}/>
        <div style={{
          position: "absolute", top: 0, left: 0,
          width: 400, height: 400,
          background: "radial-gradient(circle at top left, rgba(6,182,212,0.05), transparent 70%)",
          pointerEvents: "none",
        }}/>

        <div style={{ width: "100%", maxWidth: 380, position: "relative", zIndex: 1 }}>

          {/* logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 40 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8, background: "#06b6d4",
              display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
            }}>
              <svg width="18" height="18" viewBox="0 0 16 16" fill="none">
                <path d="M2 12L6 6L9 9L12 4L14 6" stroke="#050810" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="14" cy="6" r="1.5" fill="#050810"/>
              </svg>
            </div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "#f1f5f9", letterSpacing: "-0.01em", fontFamily: "var(--font-display, sans-serif)" }}>NeuralTrade</span>
          </div>

          {/* step indicator */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 32 }}>
            {[0, 1].map(i => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 24, height: 24, borderRadius: "50%",
                  background: i <= step ? "#06b6d4" : "rgba(255,255,255,0.05)",
                  border: `1px solid ${i <= step ? "#06b6d4" : "rgba(255,255,255,0.1)"}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10, color: i <= step ? "#050810" : "#334155",
                  fontWeight: 600, transition: "all 0.3s",
                }}>{i + 1}</div>
                <span style={{ fontSize: 10, color: i === step ? "#94a3b8" : "#334155", letterSpacing: "0.1em" }}>
                  {i === 0 ? "IDENTITY" : "PASSWORD"}
                </span>
                {i === 0 && <div style={{ width: 24, height: 1, background: step > 0 ? "#06b6d4" : "rgba(255,255,255,0.08)", transition: "background 0.3s" }}/>}
              </div>
            ))}
          </div>

          <div style={{ fontSize: 22, fontWeight: 300, color: "#f1f5f9", letterSpacing: "-0.02em", marginBottom: 4, fontFamily: "var(--font-display, sans-serif)" }}>
            {step === 0 ? "Create your account" : "Secure your account"}
          </div>
          <div style={{ fontSize: 13, color: "#475569", marginBottom: 28 }}>
            {step === 0 ? "Step 1 of 2 — your details" : "Step 2 of 2 — set a password"}
          </div>

          {/* api error */}
          {apiError && (
            <div style={{
              marginBottom: 20, padding: "11px 14px",
              background: "rgba(244,63,94,0.07)", border: "1px solid rgba(244,63,94,0.2)",
              borderRadius: 8, fontSize: 12, color: "#fca5a5",
              display: "flex", alignItems: "center", gap: 8,
            }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              {apiError}
            </div>
          )}

          {/* STEP 0 */}
          {step === 0 && (
            <form onSubmit={handleNext} noValidate>
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: "block", fontSize: 11, color: "#475569", marginBottom: 6, letterSpacing: "0.1em" }}>FULL NAME</label>
                <input type="text" autoComplete="name" placeholder="Arjun Sharma"
                  value={form.name}
                  onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                  onFocus={() => setFocusedField("name")}
                  onBlur={() => setFocusedField(null)}
                  style={inputStyle("name", !!errors.name)}
                />
                {errors.name && <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>{errors.name}</div>}
              </div>

              <div style={{ marginBottom: 28 }}>
                <label style={{ display: "block", fontSize: 11, color: "#475569", marginBottom: 6, letterSpacing: "0.1em" }}>EMAIL</label>
                <input type="email" autoComplete="email" placeholder="you@example.com"
                  value={form.email}
                  onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
                  onFocus={() => setFocusedField("email")}
                  onBlur={() => setFocusedField(null)}
                  style={inputStyle("email", !!errors.email)}
                />
                {errors.email && <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>{errors.email}</div>}
              </div>

              <button type="submit" style={{
                width: "100%", padding: 13, fontSize: 13,
                background: "#06b6d4", border: "none", borderRadius: 8,
                color: "#050810", fontWeight: 700, cursor: "pointer",
                fontFamily: "inherit", letterSpacing: "0.05em",
              }}>
                CONTINUE →
              </button>
            </form>
          )}

          {/* STEP 1 */}
          {step === 1 && (
            <form onSubmit={handleSubmit} noValidate>
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: "block", fontSize: 11, color: "#475569", marginBottom: 6, letterSpacing: "0.1em" }}>PASSWORD</label>
                <div style={{ position: "relative" }}>
                  <input
                    type={showPass ? "text" : "password"} autoComplete="new-password"
                    placeholder="Min. 8 chars, 1 uppercase, 1 number"
                    value={form.password}
                    onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
                    onFocus={() => setFocusedField("password")}
                    onBlur={() => setFocusedField(null)}
                    style={{ ...inputStyle("password", !!errors.password), paddingRight: 44 }}
                  />
                  <button type="button" onClick={() => setShowPass(p => !p)}
                    style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "#475569", padding: 4, display: "flex" }}>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      {showPass
                        ? <><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/></>
                        : <><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></>
                      }
                    </svg>
                  </button>
                </div>

                {/* strength meter */}
                {form.password && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ display: "flex", gap: 3, marginBottom: 4 }}>
                      {[1, 2, 3, 4].map(i => (
                        <div key={i} style={{
                          flex: 1, height: 3, borderRadius: 2,
                          background: i <= strength.score ? strength.color : "rgba(255,255,255,0.06)",
                          transition: "background 0.3s",
                        }}/>
                      ))}
                    </div>
                    <span style={{ fontSize: 10, color: strength.color, letterSpacing: "0.1em" }}>{strength.label}</span>
                  </div>
                )}
                {errors.password && <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>{errors.password}</div>}
              </div>

              <div style={{ marginBottom: 28 }}>
                <label style={{ display: "block", fontSize: 11, color: "#475569", marginBottom: 6, letterSpacing: "0.1em" }}>CONFIRM PASSWORD</label>
                <input
                  type="password" autoComplete="new-password" placeholder="••••••••"
                  value={form.confirm}
                  onChange={e => setForm(p => ({ ...p, confirm: e.target.value }))}
                  onFocus={() => setFocusedField("confirm")}
                  onBlur={() => setFocusedField(null)}
                  style={inputStyle("confirm", !!errors.confirm)}
                />
                {errors.confirm && <div style={{ marginTop: 4, fontSize: 11, color: "#f87171" }}>{errors.confirm}</div>}
              </div>

              <div style={{ display: "flex", gap: 10 }}>
                <button type="button" onClick={() => setStep(0)}
                  style={{
                    flex: "0 0 44px", padding: 13,
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 8, color: "#94a3b8", cursor: "pointer", fontSize: 14,
                  }}>←</button>
                <button type="submit" disabled={submitting}
                  style={{
                    flex: 1, padding: 13, fontSize: 13,
                    background: submitting ? "rgba(6,182,212,0.5)" : "#06b6d4",
                    border: "none", borderRadius: 8, color: "#050810",
                    fontWeight: 700, cursor: submitting ? "not-allowed" : "pointer",
                    fontFamily: "inherit", letterSpacing: "0.05em",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  }}>
                  {submitting ? (
                    <><span style={{ width: 14, height: 14, border: "2px solid rgba(5,8,16,0.3)", borderTopColor: "#050810", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block" }}/> CREATING...</>
                  ) : "CREATE ACCOUNT →"}
                </button>
              </div>
            </form>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: 12, margin: "24px 0" }}>
            <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }}/>
            <span style={{ fontSize: 11, color: "#334155", letterSpacing: "0.1em" }}>HAVE AN ACCOUNT?</span>
            <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.06)" }}/>
          </div>

          <Link href="/auth/login" style={{
            display: "block", width: "100%", padding: 12,
            textAlign: "center", fontSize: 13, color: "#94a3b8",
            border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8,
            textDecoration: "none", letterSpacing: "0.05em",
            background: "rgba(255,255,255,0.02)", boxSizing: "border-box",
          }}>
            SIGN IN
          </Link>

          <div style={{ marginTop: 32, fontSize: 10, color: "#1e293b", textAlign: "center", lineHeight: 1.6 }}>
            FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Serif+Display&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
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