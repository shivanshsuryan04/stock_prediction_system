import { NextRequest, NextResponse } from "next/server";

interface YahooResult {
  regularMarketPrice?: number;
  regularMarketChange?: number;
  regularMarketChangePercent?: number;
}

export async function GET(req: NextRequest) {
  const symbol = req.nextUrl.searchParams.get("symbol");
  if (!symbol) return NextResponse.json({ error: "symbol required" }, { status: 400 });

  try {
    const url =
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}` +
      `?interval=1d&range=1d`;

    const res = await fetch(url, {
      headers: {
        // Yahoo requires a valid user-agent from a browser
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
      next: { revalidate: 60 }, // ISR-style cache for 60 s
    });

    if (!res.ok) throw new Error(`Yahoo returned ${res.status}`);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const data = await res.json() as any;
    const meta = data?.chart?.result?.[0]?.meta;
    if (!meta) throw new Error("No meta in response");

    const price      = meta.regularMarketPrice ?? meta.chartPreviousClose ?? null;
    const prevClose  = meta.chartPreviousClose ?? meta.previousClose ?? price;
    const change     = (price !== null && prevClose !== null) ? +(price - prevClose).toFixed(2) : null;
    const changePercent = (price !== null && prevClose !== null && prevClose !== 0)
      ? +((price - prevClose) / prevClose * 100).toFixed(2) : null;

    return NextResponse.json({ price, change, changePercent });
  } catch (err) {
    console.error("[market/quote]", err);
    return NextResponse.json({ error: "fetch failed" }, { status: 502 });
  }
}