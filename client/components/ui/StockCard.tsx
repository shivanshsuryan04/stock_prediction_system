"use client";

import { useState } from "react";
import { SignalBadge } from "./SignalBadge";
import type { Prediction } from "@/types";
import { getSignalMeta } from "@/types";

interface StockCardProps {
  ticker: string;
  prediction: Prediction | null;
  error: string | null;
  index: number;
  onWatchlist?: boolean;
  onToggleWatchlist?: (ticker: string) => void;
}

const COMPANY_NAMES: Record<string, string> = {
  "RELIANCE.NS":  "Reliance Industries",
  "TCS.NS":       "Tata Consultancy",
  "INFY.NS":      "Infosys",
  "HDFCBANK.NS":  "HDFC Bank",
  "ICICIBANK.NS": "ICICI Bank",
  "SBIN.NS":      "State Bank of India",
  "AXISBANK.NS":  "Axis Bank",
  "WIPRO.NS":     "Wipro",
  "HCLTECH.NS":   "HCL Technologies",
  "ITC.NS":       "ITC Limited",
  "MARUTI.NS":    "Maruti Suzuki",
  "BHARTIARTL.NS":"Bharti Airtel",
};

const SECTOR: Record<string, string> = {
  "RELIANCE.NS": "ENERGY", "TCS.NS": "IT", "INFY.NS": "IT",
  "HDFCBANK.NS": "BANK", "ICICIBANK.NS": "BANK", "SBIN.NS": "BANK",
  "AXISBANK.NS": "BANK", "WIPRO.NS": "IT", "HCLTECH.NS": "IT",
  "ITC.NS": "FMCG", "MARUTI.NS": "AUTO", "BHARTIARTL.NS": "TELECOM",
};

const STAGGER = [0,60,120,180,240,300,360,420,480,540,600,660];

export function StockCard({
  ticker, prediction, error, index, onWatchlist = false, onToggleWatchlist,
}: StockCardProps) {
  const [hovered, setHovered] = useState(false);
  const delay   = STAGGER[index % STAGGER.length] ?? 0;
  const company = COMPANY_NAMES[ticker] ?? ticker;
  const sector  = SECTOR[ticker] ?? "NSE";
  const short   = ticker.replace(".NS", "");
  const meta    = prediction ? getSignalMeta(prediction.finalSignal) : null;
  const confPct = prediction ? +(prediction.lstmConf * 100).toFixed(1) : 0;

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="relative overflow-hidden animate-fade-in-up"
      style={{
        animationDelay: `${delay}ms`,
        background: "#0a0f1a",
        border: `1px solid ${hovered && meta ? meta.dotColor + "55" : "rgba(255,255,255,0.07)"}`,
        borderRadius: 4,
        transition: "border-color 0.25s, box-shadow 0.25s",
        boxShadow: hovered && meta
          ? `0 0 24px ${meta.dotColor}22, 0 0 1px ${meta.dotColor}44`
          : "none",
        fontFamily: "var(--font-mono, monospace)",
      }}
    >
      {/* left accent bar */}
      <div style={{
        position: "absolute", left: 0, top: 0, bottom: 0, width: 3,
        background: meta ? meta.dotColor : "rgba(255,255,255,0.1)",
        transition: "background 0.3s",
      }}/>

      {/* top scanline stripe */}
      <div style={{
        position: "absolute", inset: "0 0 auto 0", height: 1,
        background: meta
          ? `linear-gradient(90deg,transparent,${meta.dotColor}88,transparent)`
          : "rgba(255,255,255,0.04)",
        transition: "background 0.3s",
      }}/>

      <div style={{ padding: "14px 16px 14px 20px" }}>

        {/* ── header row ── */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 15, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.04em" }}>
                {short}
              </span>
              <span style={{
                fontSize: 9, padding: "1px 6px", borderRadius: 2,
                background: "rgba(255,255,255,0.05)", color: "#475569",
                letterSpacing: "0.15em", fontWeight: 600,
              }}>
                {sector}
              </span>
              {prediction?.fromCache && (
                <span style={{
                  fontSize: 8, padding: "1px 5px", borderRadius: 2,
                  background: "rgba(6,182,212,0.08)", color: "#06b6d4",
                  letterSpacing: "0.1em",
                }}>CACHED</span>
              )}
            </div>
            <div style={{ fontSize: 11, color: "#334155", marginTop: 2 }}>{company}</div>
          </div>

          {onToggleWatchlist && (
            <button
              onClick={() => onToggleWatchlist(ticker)}
              style={{
                background: "none", border: "none", cursor: "pointer", padding: 4,
                color: onWatchlist ? "#f59e0b" : "#1e293b",
                transition: "color 0.2s, transform 0.15s",
                transform: hovered ? "scale(1.1)" : "scale(1)",
              }}
              aria-label={onWatchlist ? `Remove ${short}` : `Watch ${short}`}
            >
              <svg width="13" height="13" viewBox="0 0 24 24"
                fill={onWatchlist ? "currentColor" : "none"}
                stroke="currentColor" strokeWidth="2">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
              </svg>
            </button>
          )}
        </div>

        {/* error */}
        {error && (
          <div style={{
            fontSize: 11, padding: "8px 10px", borderRadius: 3, marginBottom: 8,
            background: "rgba(244,63,94,0.07)", color: "#f87171",
            borderLeft: "2px solid #f43f5e",
          }}>
            {error}
          </div>
        )}

        {/* skeleton */}
        {!prediction && !error && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 4 }}>
            <div className="skeleton" style={{ height: 10, width: "70%", borderRadius: 2 }}/>
            <div className="skeleton" style={{ height: 10, width: "45%", borderRadius: 2 }}/>
            <div className="skeleton" style={{ height: 28, width: "100%", borderRadius: 2, marginTop: 6 }}/>
          </div>
        )}

        {/* prediction data */}
        {prediction && meta && (
          <>
            {/* ── model signals row ── */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr",
              gap: 8, marginBottom: 12,
            }}>
              {[
                { label: "XGBOOST", sig: prediction.xgbSignal },
                { label: "LSTM",    sig: prediction.lstmSignal },
              ].map(({ label, sig }) => (
                <div key={label} style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.05)",
                  borderRadius: 3, padding: "7px 9px",
                }}>
                  <div style={{ fontSize: 8, color: "#334155", letterSpacing: "0.15em", marginBottom: 5 }}>
                    {label}
                  </div>
                  <SignalBadge signal={sig} size="sm" />
                </div>
              ))}
            </div>

            {/* ── LSTM confidence ── */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <span style={{ fontSize: 8, color: "#334155", letterSpacing: "0.15em" }}>LSTM CONF</span>
                <span style={{ fontSize: 10, color: meta.dotColor, fontWeight: 700 }}>{confPct}%</span>
              </div>
              {/* segmented bar */}
              <div style={{ display: "flex", gap: 2 }}>
                {Array.from({ length: 20 }).map((_, i) => (
                  <div key={i} style={{
                    flex: 1, height: 4, borderRadius: 1,
                    background: i < Math.round(confPct / 5)
                      ? meta.dotColor
                      : "rgba(255,255,255,0.05)",
                    transition: `background 0.05s ${i * 30}ms`,
                  }}/>
                ))}
              </div>
            </div>

            {/* ── ensemble final signal ── */}
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "9px 10px", borderRadius: 3,
              background: meta.bgStyle,
              border: `1px solid ${meta.dotColor}33`,
            }}>
              <span style={{ fontSize: 8, color: "#475569", letterSpacing: "0.2em" }}>ENSEMBLE</span>
              <SignalBadge signal={prediction.finalSignal} size="md" />
            </div>

            {/* ── footer ── */}
            <div style={{
              display: "flex", justifyContent: "space-between",
              marginTop: 10, fontSize: 9, color: "#1e293b",
            }}>
              <span>NSE · {new Date(prediction.cachedAt).toLocaleDateString("en-IN")}</span>
              <span>{new Date(prediction.cachedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}