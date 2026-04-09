import { getSignalMeta } from "@/types";
import clsx from "clsx";

interface SignalBadgeProps {
  signal: string;
  size?: "sm" | "md" | "lg";
}

const ICONS: Record<string, string> = {
  "Strong Buy":  "▲▲",
  "Buy":         "▲",
  "Hold":        "◆",
  "Sell":        "▼",
  "Strong Sell": "▼▼",
};

export function SignalBadge({ signal, size = "md" }: SignalBadgeProps) {
  const meta = getSignalMeta(signal);
  const icon = ICONS[meta.label] ?? "◆";
  const isBull = meta.direction === "bull";
  const isBear = meta.direction === "bear";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 font-bold tracking-widest uppercase",
        "border-l-2",
        {
          "text-[9px] px-1.5 py-0.5":  size === "sm",
          "text-[10px] px-2.5 py-1":   size === "md",
          "text-xs px-3 py-1.5":       size === "lg",
        }
      )}
      style={{
        background:   meta.bgStyle,
        color:        meta.dotColor,
        borderColor:  meta.dotColor,
        borderRadius: "2px",
        letterSpacing: "0.12em",
        boxShadow: `inset 0 0 12px ${meta.dotColor}18`,
      }}
    >
      {/* animated live dot for bull/bear */}
      {(isBull || isBear) && (
        <span
          className="shrink-0 rounded-full"
          style={{
            width: size === "sm" ? 4 : 5,
            height: size === "sm" ? 4 : 5,
            background: meta.dotColor,
            animation: "badgePulse 1.8s ease-in-out infinite",
          }}
        />
      )}
      <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "inherit" }}>
        {icon} {meta.label}
      </span>
      <style>{`@keyframes badgePulse{0%,100%{opacity:1}50%{opacity:0.35}}`}</style>
    </span>
  );
}