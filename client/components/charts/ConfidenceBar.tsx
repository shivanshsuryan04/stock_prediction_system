"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictionEntry } from "@/types";
import { getSignalMeta } from "@/types";

interface ConfidenceBarProps {
  predictions: PredictionEntry[];
  limit?: number;
  showSignal?: boolean;
  animate?: boolean;
}

interface BarItem {
  ticker: string;
  confidence: number;
  signal: string;
  dotColor: string;
  bgStyle: string;
  borderStyle: string;
  direction: string;
}

export function ConfidenceBar({
  predictions, limit, showSignal = true, animate = true,
}: ConfidenceBarProps) {
  const [mounted, setMounted] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const bars: BarItem[] = predictions
    .filter((p) => p.data !== null)
    .map((p) => {
      const meta = getSignalMeta(p.data!.finalSignal);
      return {
        ticker: p.ticker.replace(".NS", ""),
        confidence: p.data!.lstmConf,
        signal: meta.label,
        dotColor: meta.dotColor,
        bgStyle: meta.bgStyle,
        borderStyle: meta.borderStyle,
        direction: meta.direction,
      };
    })
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, limit ?? predictions.length);

  if (bars.length === 0) return null;

  return (
    <div
      ref={containerRef}
      style={{
        background: "#0a0f1a",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 4,
        padding: "20px 24px",
        fontFamily: "var(--font-mono, monospace)",
      }}
    >
      {/* header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 9, color: "#334155", letterSpacing: "0.2em", marginBottom: 4 }}>
            MODEL OUTPUT
          </div>
          <div style={{ fontSize: 14, fontWeight: 700, color: "#f1f5f9", letterSpacing: "0.04em" }}>
            LSTM Confidence Ranking
          </div>
        </div>
        <div style={{ display: "flex", gap: 16, fontSize: 9, color: "#1e293b" }}>
          <span>0%</span>
          <span>25%</span>
          <span>50%</span>
          <span>75%</span>
          <span>100%</span>
        </div>
      </div>

      {/* 50% guide line */}
      <div style={{ position: "relative" }}>
        <div style={{
          position: "absolute",
          left: `calc(72px + (100% - 72px - ${showSignal ? "90px" : "0px"}) * 0.5)`,
          top: 0, bottom: 0, width: 1,
          background: "rgba(255,255,255,0.04)",
          pointerEvents: "none",
          zIndex: 0,
        }}/>

        {/* bars */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {bars.map((bar, idx) => {
            const pct      = +(bar.confidence * 100).toFixed(1);
            const barWidth = mounted || !animate ? `${pct}%` : "0%";
            const isTop3   = idx < 3;

            return (
              <div key={bar.ticker} style={{ display: "flex", alignItems: "center", gap: 30, position: "relative", zIndex: 1 }}>

                {/* rank + ticker */}
                <div style={{ display: "flex", alignItems: "center", gap: 6, width: 72, flexShrink: 0 }}>
                  <span style={{
                    fontSize: 9, width: 16, color: isTop3 ? bar.dotColor : "#1e293b",
                    fontWeight: 700, textAlign: "right", flexShrink: 0,
                  }}>
                    #{idx + 1}
                  </span>
                  <span style={{
                    fontSize: 11, fontWeight: 700,
                    color: isTop3 ? "#94a3b8" : "#334155",
                    letterSpacing: "0.06em",
                  }}>
                    {bar.ticker}
                  </span>
                </div>

                {/* track */}
                <div style={{
                  flex: 1, height: 30, position: "relative",
                  background: "rgba(255,255,255,0.03)",
                  borderRadius: 2, overflow: "hidden",
                  border: "1px solid rgba(255,255,255,0.04)",
                }}>
                  {/* fill */}
                  <div style={{
                    position: "absolute", inset: "0 auto 0 0",
                    width: barWidth,
                    background: bar.dotColor,
                    opacity: 0.75,
                    borderRadius: 2,
                    transition: animate
                      ? `width 0.8s cubic-bezier(0.22,1,0.36,1) ${idx * 45}ms`
                      : "none",
                  }}/>
                  {/* pct label */}
                  {bar.confidence > 0.15 && (
                    <span style={{
                      position: "absolute", left: 8, top: "50%",
                      transform: "translateY(-50%)",
                      fontSize: 9, fontWeight: 700,
                      color: "#050810",
                      mixBlendMode: "luminosity",
                      pointerEvents: "none",
                      letterSpacing: "0.05em",
                    }}>
                      {pct}%
                    </span>
                  )}
                </div>

                {/* signal pill */}
                {showSignal && (
                  <div style={{
                    width: 82, flexShrink: 0, textAlign: "center",
                    fontSize: 9, fontWeight: 700,
                    color: bar.dotColor,
                    padding: "3px 0",
                    background: bar.bgStyle,
                    borderLeft: `2px solid ${bar.dotColor}`,
                    borderRadius: "0 2px 2px 0",
                    letterSpacing: "0.1em",
                  }}>
                    {bar.signal.toUpperCase()}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 14, fontSize: 9, color: "#1e293b", letterSpacing: "0.08em" }}>
        VALUES NEAR 0.50 INDICATE MODEL UNCERTAINTY · REFRESHED EVERY 30 MIN
      </div>
    </div>
  );
}