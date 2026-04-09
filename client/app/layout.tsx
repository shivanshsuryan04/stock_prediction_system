import type { Metadata } from "next";
import { Space_Grotesk, DM_Mono } from "next/font/google";
import "./globals.css";
import { AuthHydrator } from "@/components/layout/AuthHydrator";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["300", "400", "500", "600", "700"],
});

const dmMono = DM_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["300", "400", "500"],
});

export const metadata: Metadata = {
  title: "NeuralTrade | AI Stock Intelligence",
  description: "Institutional-grade AI stock predictions powered by XGBoost + LSTM ensemble models",
  keywords: ["stock prediction", "AI trading", "NSE stocks", "machine learning finance"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${dmMono.variable}`}>
      <body className="bg-neutral-950 text-neutral-100 antialiased">
        {/* Hydrates auth state from refresh token cookie on every page load */}
        <AuthHydrator />
        {children}
      </body>
    </html>
  );
}