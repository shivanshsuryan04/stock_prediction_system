"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Navbar } from "@/components/layout/Navbar";

const NAV_ITEMS = [
  {
    href: "/dashboard",
    label: "DASHBOARD",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
        <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
      </svg>
    ),
  },
  {
    href: "/dashboard/analytics",
    label: "ANALYTICS",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
  },
  {
    href: "/dashboard/screener",
    label: "SCREENER",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    ),
  },
  {
    href: "/dashboard/watchlist",
    label: "WATCHLIST",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
      </svg>
    ),
  },
  {
    href: "/dashboard/portfolio",
    label: "PORTFOLIO", 
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
  },
  {
    href: "/dashboard/alerts",
    label: "ALERTS",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
    ),
  },
  {
    href: "/dashboard/settings",
    label: "SETTINGS",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    ),
  },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  // Exact match for /dashboard, prefix match for sub-pages
  function isActive(href: string) {
    if (href === "/dashboard") return pathname === "/dashboard";
    return pathname.startsWith(href);
  }

  return (
    <div style={{ minHeight: "100vh", background: "#050810", fontFamily: "var(--font-mono, monospace)" }}>
      <Navbar />

      <div style={{ display: "flex", maxWidth: 1280, margin: "0 auto", padding: "0 24px" }}>
        {/* Sidebar */}
        <aside style={{
          width: 200, flexShrink: 0, paddingTop: 24,
          position: "sticky", top: 52, height: "calc(100vh - 52px)",
          overflowY: "auto", paddingRight: 16,
        }}>
          <div style={{ marginBottom: 6, fontSize: 8, color: "#1e293b", letterSpacing: "0.2em", paddingLeft: 10 }}>
            MAIN
          </div>

          {NAV_ITEMS.slice(0, 5).map((item) => (
            <Link key={item.href} href={item.href} style={{ textDecoration: "none" }}>
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "7px 10px", borderRadius: 4, marginBottom: 2,
                background: isActive(item.href) ? "rgba(6,182,212,0.1)" : "transparent",
                border: isActive(item.href) ? "1px solid rgba(6,182,212,0.2)" : "1px solid transparent",
                color: isActive(item.href) ? "#06b6d4" : "#334155",
                fontSize: 10, letterSpacing: "0.08em",
                transition: "all 0.15s",
                cursor: "pointer",
              }}
                onMouseEnter={e => {
                  if (!isActive(item.href)) {
                    (e.currentTarget as HTMLDivElement).style.background = "rgba(255,255,255,0.03)";
                    (e.currentTarget as HTMLDivElement).style.color = "#94a3b8";
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive(item.href)) {
                    (e.currentTarget as HTMLDivElement).style.background = "transparent";
                    (e.currentTarget as HTMLDivElement).style.color = "#334155";
                  }
                }}
              >
                <span style={{ opacity: isActive(item.href) ? 1 : 0.6, flexShrink: 0 }}>{item.icon}</span>
                {item.label}
              </div>
            </Link>
          ))}

          <div style={{ marginTop: 16, marginBottom: 6, fontSize: 8, color: "#1e293b", letterSpacing: "0.2em", paddingLeft: 10 }}>
            TOOLS
          </div>

          {NAV_ITEMS.slice(5).map((item) => (
            <Link key={item.href} href={item.href} style={{ textDecoration: "none" }}>
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                padding: "7px 10px", borderRadius: 4, marginBottom: 2,
                background: isActive(item.href) ? "rgba(6,182,212,0.1)" : "transparent",
                border: isActive(item.href) ? "1px solid rgba(6,182,212,0.2)" : "1px solid transparent",
                color: isActive(item.href) ? "#06b6d4" : "#334155",
                fontSize: 10, letterSpacing: "0.08em",
                transition: "all 0.15s",
                cursor: "pointer",
              }}>
                <span style={{ opacity: isActive(item.href) ? 1 : 0.6, flexShrink: 0 }}>{item.icon}</span>
                {item.label}
              </div>
            </Link>
          ))}

          {/* Live status */}
          <div style={{ position: "absolute", bottom: 20, left: 0, right: 16, padding: "10px 12px", background: "#080d18", borderRadius: 4, border: "1px solid rgba(255,255,255,0.05)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block", animation: "statusPulse 2s ease-in-out infinite" }} />
              <span style={{ fontSize: 9, color: "#1e293b", letterSpacing: "0.12em" }}>NSE LIVE</span>
            </div>
            <div style={{ fontSize: 9, color: "#1e293b" }}>
              {new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })} IST
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main style={{ flex: 1, paddingTop: 24, paddingBottom: 20, minWidth: 0 }}>
          {children}
        </main>
      </div>

      <style>{`
        @keyframes statusPulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}