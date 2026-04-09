"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/lib/auth.store";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();
  const router   = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (isLoading) return;
    const isAuthPage = pathname.startsWith("/auth");
    if (!isAuthenticated && !isAuthPage)  router.replace("/auth/login");
    if (isAuthenticated  && isAuthPage)   router.replace("/dashboard");
  }, [isAuthenticated, isLoading, router, pathname]);

  if (isLoading) {
    return (
      <div style={{
        minHeight: "100vh", display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        background: "#050810",
        fontFamily: "var(--font-mono, monospace)",
      }}>
        {/* logo mark */}
        <div style={{
          width: 40, height: 40, borderRadius: 6,
          background: "#06b6d4",
          display: "flex", alignItems: "center", justifyContent: "center",
          marginBottom: 24,
          animation: "authLogoPulse 1.5s ease-in-out infinite",
        }}>
          <svg width="20" height="20" viewBox="0 0 16 16" fill="none">
            <path d="M2 12L6 6L9 9L12 4L14 6" stroke="#050810" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="14" cy="6" r="1.5" fill="#050810"/>
          </svg>
        </div>

        {/* segmented progress bar */}
        <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} style={{
              width: 24, height: 3, borderRadius: 2,
              background: "#06b6d4",
              animation: `segLoad 1.2s ease-in-out infinite`,
              animationDelay: `${i * 0.12}s`,
              opacity: 0.2,
            }}/>
          ))}
        </div>

        <div style={{ fontSize: 10, color: "#334155", letterSpacing: "0.2em" }}>
          VERIFYING SESSION
        </div>

        <style>{`
          @keyframes authLogoPulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
          @keyframes segLoad { 0%,100%{opacity:0.2} 50%{opacity:1} }
        `}</style>
      </div>
    );
  }

  return <>{children}</>;
}