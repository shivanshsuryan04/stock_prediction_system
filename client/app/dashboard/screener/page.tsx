"use client";

import { useState, useMemo } from "react";
import { usePredictions } from "@/hooks/usePredictions";
import { useWatchlist } from "@/hooks/useWatchlist";
import { SignalBadge } from "@/components/ui/SignalBadge";
import { getSignalMeta } from "@/types";

type SortKey = "ticker" | "conf" | "signal";
type SortDir = "asc" | "desc";

const SECTOR: Record<string, string> = {
  "RELIANCE.NS":"ENERGY","TCS.NS":"IT","INFY.NS":"IT",
  "HDFCBANK.NS":"BANK","ICICIBANK.NS":"BANK","SBIN.NS":"BANK",
  "AXISBANK.NS":"BANK","WIPRO.NS":"IT","HCLTECH.NS":"IT",
  "ITC.NS":"FMCG","MARUTI.NS":"AUTO","BHARTIARTL.NS":"TELECOM",
};
const COMPANY: Record<string, string> = {
  "RELIANCE.NS":"Reliance Industries","TCS.NS":"Tata Consultancy","INFY.NS":"Infosys",
  "HDFCBANK.NS":"HDFC Bank","ICICIBANK.NS":"ICICI Bank","SBIN.NS":"State Bank",
  "AXISBANK.NS":"Axis Bank","WIPRO.NS":"Wipro","HCLTECH.NS":"HCL Technologies",
  "ITC.NS":"ITC Limited","MARUTI.NS":"Maruti Suzuki","BHARTIARTL.NS":"Bharti Airtel",
};
const ALL_SECTORS = ["IT","BANK","ENERGY","AUTO","FMCG","TELECOM"];

// Signal buckets — we match by UPPERCASE contains, so backend casing doesn't matter
type SigBucket = "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL";
const SIG_BUCKETS: SigBucket[] = ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"];

function getSigBucket(sig: string): SigBucket {
  const s = (sig ?? "").toUpperCase();
  if (s.includes("STRONG BUY"))  return "STRONG BUY";
  if (s.includes("BUY"))         return "BUY";
  if (s.includes("STRONG SELL")) return "STRONG SELL";
  if (s.includes("SELL"))        return "SELL";
  return "HOLD";
}

const SIG_COLORS: Record<SigBucket, string> = {
  "STRONG BUY":"#06b6d4","BUY":"#10b981","HOLD":"#f59e0b","SELL":"#f43f5e","STRONG SELL":"#ef4444",
};

export default function ScreenerPage() {
  const { predictions, isLoading } = usePredictions();
  const { isOnWatchlist, addTicker, removeTicker } = useWatchlist();

  const [minConf, setMinConf]     = useState(0);
  const [sigFilter, setSigFilter] = useState<Set<SigBucket>>(new Set(SIG_BUCKETS));
  const [secFilter, setSecFilter] = useState<Set<string>>(new Set(ALL_SECTORS));
  const [sortKey, setSortKey]     = useState<SortKey>("conf");
  const [sortDir, setSortDir]     = useState<SortDir>("desc");
  const [search, setSearch]       = useState("");

  const withData = predictions.filter(p => p.data !== null);

  const toggleSig = (s: SigBucket) =>
    setSigFilter(prev => { const n = new Set(prev); n.has(s)?n.delete(s):n.add(s); return n; });
  const toggleSec = (s: string) =>
    setSecFilter(prev => { const n = new Set(prev); n.has(s)?n.delete(s):n.add(s); return n; });

  function handleSort(key: SortKey) {
    if (sortKey===key) setSortDir(d=>d==="asc"?"desc":"asc");
    else { setSortKey(key); setSortDir("desc"); }
  }

  const filtered = useMemo(() => {
    return withData
      .filter(p => {
        const sec    = SECTOR[p.ticker]??"OTHER";
        const bucket = getSigBucket(p.data!.finalSignal);
        const conf   = (p.data!.lstmConf??0)*100;
        return (
          sigFilter.has(bucket) &&
          secFilter.has(sec) &&
          conf >= minConf &&
          (!search || p.ticker.toLowerCase().includes(search.toLowerCase()) ||
           (COMPANY[p.ticker]??"").toLowerCase().includes(search.toLowerCase()))
        );
      })
      .sort((a,b) => {
        let va: number|string=0, vb: number|string=0;
        if (sortKey==="conf")   { va=a.data!.lstmConf??0; vb=b.data!.lstmConf??0; }
        if (sortKey==="ticker") { va=a.ticker; vb=b.ticker; }
        if (sortKey==="signal") { va=a.data!.finalSignal??""; vb=b.data!.finalSignal??""; }
        if (typeof va==="string")
          return sortDir==="asc"?va.localeCompare(vb as string):(vb as string).localeCompare(va);
        return sortDir==="asc"?(va as number)-(vb as number):(vb as number)-(va as number);
      });
  }, [withData, sigFilter, secFilter, minConf, search, sortKey, sortDir]);

  const mono: React.CSSProperties = { fontFamily:"var(--font-mono, monospace)" };

  return (
    <div style={mono}>
      {/* Header */}
      <div style={{ marginBottom:24, paddingBottom:18, borderBottom:"1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:6 }}>SIGNAL SCANNER</div>
        <h1 style={{ fontSize:22, fontWeight:700, color:"#f1f5f9", margin:0 }}>
          Stock<span style={{ color:"#06b6d4" }}>_</span>Screener
        </h1>
        <p style={{ fontSize:11, color:"#334155", marginTop:4 }}>
          {isLoading ? "Loading signals..." : `${filtered.length} of ${withData.length} stocks match your filters`}
        </p>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"220px 1fr", gap:14 }}>
        {/* Filters sidebar */}
        <div>
          {/* Search */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:12, marginBottom:10 }}>
            <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:8 }}>SEARCH</div>
            <input
              type="text" placeholder="TICKER OR NAME..." value={search}
              onChange={e=>setSearch(e.target.value)}
              style={{ background:"#050810", border:"1px solid rgba(255,255,255,0.08)", borderRadius:3, padding:"6px 10px", fontSize:10, color:"#94a3b8", fontFamily:"inherit", letterSpacing:"0.06em", outline:"none", width:"100%", boxSizing:"border-box" }}
            />
          </div>

          {/* Signal filter */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:12, marginBottom:10 }}>
            <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:8 }}>SIGNAL</div>
            {SIG_BUCKETS.map(s => (
              <label key={s} style={{ display:"flex", alignItems:"center", gap:8, cursor:"pointer", fontSize:10, color: sigFilter.has(s)?SIG_COLORS[s]:"#334155", padding:"5px 0", transition:"color .15s" }}>
                <input type="checkbox" checked={sigFilter.has(s)} onChange={()=>toggleSig(s)}
                  style={{ accentColor:SIG_COLORS[s] }} />
                {s}
              </label>
            ))}
          </div>

          {/* Confidence slider */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:12, marginBottom:10 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
              <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em" }}>MIN CONF</div>
              <div style={{ fontSize:9, color:"#06b6d4", fontWeight:700 }}>{minConf}%</div>
            </div>
            <input type="range" min="0" max="100" step="5" value={minConf}
              onChange={e=>setMinConf(+e.target.value)}
              style={{ width:"100%", accentColor:"#06b6d4" }} />
          </div>

          {/* Sector filter */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:12 }}>
            <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:8 }}>SECTOR</div>
            {ALL_SECTORS.map(s => (
              <label key={s} style={{ display:"flex", alignItems:"center", gap:8, cursor:"pointer", fontSize:10, color:secFilter.has(s)?"#94a3b8":"#334155", padding:"4px 0", transition:"color .15s" }}>
                <input type="checkbox" checked={secFilter.has(s)} onChange={()=>toggleSec(s)}
                  style={{ accentColor:"#06b6d4" }} />
                {s}
              </label>
            ))}
          </div>
        </div>

        {/* Results table */}
        <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, overflow:"hidden" }}>
          {/* Header */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 100px 110px 110px 80px 36px", alignItems:"center", padding:"8px 16px", background:"rgba(255,255,255,0.02)", borderBottom:"1px solid rgba(255,255,255,0.05)" }}>
            <button onClick={()=>handleSort("ticker")} style={{ ...mono, background:"none", border:"none", cursor:"pointer", fontSize:8, color:sortKey==="ticker"?"#06b6d4":"#1e293b", letterSpacing:"0.18em", textAlign:"left" }}>
              STOCK {sortKey==="ticker"?(sortDir==="asc"?"↑":"↓"):""}
            </button>
            <span style={{ fontSize:8, color:"#1e293b", letterSpacing:"0.18em" }}>XGBOOST</span>
            <span style={{ fontSize:8, color:"#1e293b", letterSpacing:"0.18em" }}>LSTM</span>
            <span style={{ fontSize:8, color:"#1e293b", letterSpacing:"0.18em" }}>ENSEMBLE</span>
            <button onClick={()=>handleSort("conf")} style={{ ...mono, background:"none", border:"none", cursor:"pointer", fontSize:8, color:sortKey==="conf"?"#06b6d4":"#1e293b", letterSpacing:"0.18em" }}>
              CONF {sortKey==="conf"?(sortDir==="asc"?"↑":"↓"):""}
            </button>
            <span />
          </div>

          {isLoading && (
            <div style={{ padding:32, textAlign:"center", fontSize:10, color:"#334155", letterSpacing:"0.15em" }}>
              LOADING SIGNALS...
            </div>
          )}

          {!isLoading && filtered.length===0 && (
            <div style={{ padding:"60px 24px", textAlign:"center" }}>
              <div style={{ fontSize:10, color:"#334155", letterSpacing:"0.15em", marginBottom:12 }}>NO STOCKS MATCH YOUR FILTERS</div>
              <button onClick={()=>{ setSigFilter(new Set(SIG_BUCKETS)); setSecFilter(new Set(ALL_SECTORS)); setMinConf(0); setSearch(""); }}
                style={{ ...mono, background:"none", border:"1px solid rgba(255,255,255,0.08)", borderRadius:3, padding:"6px 16px", fontSize:10, color:"#475569", cursor:"pointer", letterSpacing:"0.1em" }}>
                CLEAR FILTERS
              </button>
            </div>
          )}

          {!isLoading && filtered.map((p, idx) => {
            const meta    = getSignalMeta(p.data!.finalSignal);
            const confPct = Math.round((p.data!.lstmConf??0)*100);
            const onWatch = isOnWatchlist(p.ticker);

            return (
              <div key={p.ticker} style={{
                display:"grid", gridTemplateColumns:"1fr 100px 110px 110px 80px 36px",
                alignItems:"center", padding:"10px 16px",
                borderBottom: idx<filtered.length-1?"1px solid rgba(255,255,255,0.04)":"none",
                borderLeft:`3px solid ${meta.dotColor}66`,
                animation:"fadeUp 0.25s ease both",
                animationDelay:`${Math.min(idx*25,300)}ms`,
              }}>
                {/* Stock info */}
                <div>
                  <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                    <span style={{ fontSize:12, fontWeight:700, color:"#f1f5f9", letterSpacing:"0.04em" }}>{p.ticker.replace(".NS","")}</span>
                    <span style={{ fontSize:8, padding:"1px 5px", borderRadius:2, background:"rgba(255,255,255,0.05)", color:"#475569", letterSpacing:"0.12em" }}>{SECTOR[p.ticker]??"NSE"}</span>
                    {p.data!.fromCache && <span style={{ fontSize:8, padding:"1px 5px", borderRadius:2, background:"rgba(6,182,212,0.08)", color:"#06b6d4", letterSpacing:"0.1em" }}>CACHED</span>}
                  </div>
                  <div style={{ fontSize:10, color:"#334155", marginTop:1 }}>{COMPANY[p.ticker]??p.ticker}</div>
                </div>
                <div><SignalBadge signal={p.data!.xgbSignal} size="sm" /></div>
                <div>
                  <SignalBadge signal={p.data!.lstmSignal} size="sm" />
                  <div style={{ fontSize:9, color:"#334155", marginTop:2 }}>{confPct}% conf</div>
                </div>
                <div><SignalBadge signal={p.data!.finalSignal} size="sm" /></div>
                {/* Conf bar */}
                <div>
                  <div style={{ fontSize:9, fontWeight:700, color:meta.dotColor, marginBottom:2 }}>{confPct}%</div>
                  <div style={{ height:3, background:"rgba(255,255,255,0.05)", borderRadius:2, overflow:"hidden" }}>
                    <div style={{ height:"100%", width:`${confPct}%`, background:meta.dotColor, borderRadius:2 }} />
                  </div>
                </div>
                {/* Watchlist toggle */}
                <button
                  onClick={()=>onWatch?void removeTicker(p.ticker):void addTicker(p.ticker)}
                  style={{ background:"none", border:"none", cursor:"pointer", color:onWatch?"#f59e0b":"#1e293b", fontSize:15, padding:4, transition:"color .2s" }}
                  onMouseEnter={e=>(e.currentTarget.style.color="#f59e0b")}
                  onMouseLeave={e=>(e.currentTarget.style.color=onWatch?"#f59e0b":"#1e293b")}
                  title={onWatch?"Remove from watchlist":"Add to watchlist"}
                >★</button>
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop:20, padding:"10px 14px", background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.05)", borderLeft:"2px solid #334155", borderRadius:"0 4px 4px 0", fontSize:9, color:"#1e293b", letterSpacing:"0.08em",textAlign: "center",}}>
        AI SIGNALS ARE FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
      </div>

      <style>{`@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}`}</style>
    </div>
  );
}