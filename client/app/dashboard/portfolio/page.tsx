"use client";

import { useMemo } from "react";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  ArcElement, Tooltip, Filler,
} from "chart.js";
import { Line, Doughnut } from "react-chartjs-2";
import { useWatchlist } from "@/hooks/useWatchlist";
import { usePredictions } from "@/hooks/usePredictions";
import { SignalBadge } from "@/components/ui/SignalBadge";
import { getSignalMeta } from "@/types";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, ArcElement, Tooltip, Filler);

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

// Deterministic mock entry prices (seeded by ticker so consistent per session)
function mockEntry(ticker: string) {
  const seed = ticker.split("").reduce((a,c)=>a+c.charCodeAt(0),0);
  return 500 + (seed % 2000);
}
function mockCmp(entry: number, lstmConf: number, isBull: boolean) {
  // CMP is biased by signal direction
  const pct = isBull ? 0.05 + lstmConf*0.1 : -(0.03 + lstmConf*0.08);
  return +(entry*(1+pct)).toFixed(2);
}

const GRID="#rgba(255,255,255,0.04)";
const TICK="#334155";
const TF={ size:9 } as const;

const SECTOR_COLORS: Record<string,string> = {
  IT:"#06b6d4", BANK:"#10b981", ENERGY:"#f59e0b",
  AUTO:"#f43f5e", FMCG:"#8b5cf6", TELECOM:"#475569",
};

export default function PortfolioPage() {
  const { items, isLoading: wlLoading } = useWatchlist();
  const { predictions, isLoading: predLoading } = usePredictions();
  const isLoading = wlLoading || predLoading;

  // Build portfolio rows from watchlist items + predictions
  const rows = useMemo(() => {
    return items.map(item => {
      // find prediction from either watchlist prediction or main predictions list
      const pred = item.prediction
        ?? predictions.find(p=>p.ticker===item.ticker)?.data
        ?? null;
      const meta    = pred ? getSignalMeta(pred.finalSignal) : null;
      const isBull  = meta?.direction === "bull";
      const conf    = pred?.lstmConf ?? 0.5;
      const entry   = mockEntry(item.ticker);
      const cmp     = pred ? mockCmp(entry, conf, isBull) : entry;
      const pnlPct  = +((cmp-entry)/entry*100).toFixed(2);
      const pnlAmt  = +(cmp-entry).toFixed(2);
      return { item, pred, meta, entry, cmp, pnlPct, pnlAmt, sec: SECTOR[item.ticker]??"NSE" };
    });
  }, [items, predictions]);

  // Portfolio summary
  const totalValue    = rows.reduce((s,r)=>s+r.cmp*1,0); // treat qty=1 for display
  const totalPnl      = rows.reduce((s,r)=>s+r.pnlAmt,0);
  const bullRows      = rows.filter(r=>r.meta?.direction==="bull").length;
  const winRate       = rows.length ? Math.round(rows.filter(r=>r.pnlPct>0).length/rows.length*100) : 0;

  // Sector allocation for donut
  const sectorMap = useMemo(()=>{
    const m: Record<string,number>={};
    rows.forEach(r=>{ m[r.sec]=(m[r.sec]??0)+1; });
    return m;
  },[rows]);
  const sectorLabels = Object.keys(sectorMap);
  const sectorVals   = sectorLabels.map(s=>sectorMap[s]);
  const sectorCols   = sectorLabels.map(s=>SECTOR_COLORS[s]??"#475569");

  // Mock portfolio value history (20 days)
  const histLabels = Array.from({length:20},(_,i)=>`D${i+1}`);
  const baseVal    = Math.max(rows.length*800, 5000);
  const histVals   = histLabels.map((_,i)=>+(baseVal+i*baseVal*0.008+Math.random()*baseVal*0.01).toFixed(0));

  const mono: React.CSSProperties = { fontFamily:"var(--font-mono, monospace)" };

  function StatCard({ label, value, sub, color }: { label:string; value:string; sub:string; color:string }) {
    return (
      <div style={{ background:"#0a0f1a", border:`1px solid ${color}22`, borderTop:`2px solid ${color}`, borderRadius:4, padding:"12px 14px" }}>
        <div style={{ fontSize:8, color:"#334155", letterSpacing:"0.2em", marginBottom:6 }}>{label}</div>
        <div style={{ fontSize:20, fontWeight:700, color, lineHeight:1, marginBottom:3 }}>{value}</div>
        <div style={{ fontSize:9, color:"#1e293b" }}>{sub}</div>
      </div>
    );
  }

  return (
    <div style={mono}>
      {/* Header */}
      <div style={{ marginBottom:24, paddingBottom:18, borderBottom:"1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:6 }}>WATCHLIST PORTFOLIO</div>
        <h1 style={{ fontSize:22, fontWeight:700, color:"#f1f5f9", margin:0 }}>
          Portfolio<span style={{ color:"#06b6d4" }}>_</span>Tracker
        </h1>
        <p style={{ fontSize:11, color:"#334155", marginTop:4 }}>
          {isLoading ? "Loading..." : `${rows.length} stocks from your watchlist`}
        </p>
      </div>

      {/* Empty watchlist state */}
      {!isLoading && rows.length===0 && (
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:"80px 0", textAlign:"center", background:"#0a0f1a", border:"1px dashed rgba(255,255,255,0.06)", borderRadius:4 }}>
          <div style={{ fontSize:11, color:"#334155", letterSpacing:"0.15em", marginBottom:8 }}>NO STOCKS IN WATCHLIST</div>
          <div style={{ fontSize:10, color:"#1e293b" }}>Add stocks from the Watchlist page to track them here</div>
        </div>
      )}

      {/* Summary cards */}
      {!isLoading && rows.length>0 && (
        <>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8, marginBottom:20 }}>
            <StatCard label="POSITIONS"     value={String(rows.length)}               sub="In your watchlist"    color="#06b6d4" />
            <StatCard label="TOTAL P&L"     value={`${totalPnl>=0?"+":""}₹${Math.abs(totalPnl).toLocaleString("en-IN",{maximumFractionDigits:0})}`} sub="Simulated from signals" color={totalPnl>=0?"#10b981":"#f43f5e"} />
            <StatCard label="BULLISH"       value={String(bullRows)}                  sub="Buy signal positions"  color="#10b981" />
            <StatCard label="WIN RATE"      value={`${winRate}%`}                     sub="Positions in profit"  color="#8b5cf6" />
          </div>

          {/* Portfolio value chart */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14, marginBottom:16 }}>
            <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12 }}>
              <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em" }}>SIMULATED PORTFOLIO VALUE (20-DAY)</div>
              <div style={{ fontSize:9, color:"#10b981", fontWeight:700 }}>▲ SIGNAL-BASED GROWTH</div>
            </div>
            <div style={{ height:160 }}>
              <Line
                data={{ labels:histLabels, datasets:[{
                  data:histVals,
                  borderColor:"#10b981", backgroundColor:"rgba(16,185,129,0.06)",
                  fill:true, tension:.4, borderWidth:2, pointRadius:0,
                }]}}
                options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{
                  x:{grid:{color:"rgba(255,255,255,0.04)"},ticks:{color:TICK,font:TF,maxTicksLimit:6}},
                  y:{grid:{color:"rgba(255,255,255,0.04)"},ticks:{color:TICK,font:TF,callback:(v)=>`₹${(+v).toLocaleString("en-IN")}`}},
                }}}
              />
            </div>
          </div>

          {/* Holdings table */}
          <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, overflow:"hidden", marginBottom:16 }}>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 90px 80px 80px 80px 80px", padding:"7px 16px", background:"rgba(255,255,255,0.02)", borderBottom:"1px solid rgba(255,255,255,0.05)", fontSize:8, color:"#1e293b", letterSpacing:"0.15em" }}>
              <span>STOCK</span><span>SIGNAL</span><span style={{textAlign:"right"}}>ENTRY</span><span style={{textAlign:"right"}}>CMP</span><span style={{textAlign:"right"}}>P&L</span><span style={{textAlign:"right"}}>CONF</span>
            </div>
            {rows.map((r, idx)=>{
              const meta2 = r.meta ?? getSignalMeta("Hold");
              return (
                <div key={r.item.ticker} style={{
                  display:"grid", gridTemplateColumns:"1fr 90px 80px 80px 80px 80px",
                  alignItems:"center", padding:"10px 16px",
                  borderBottom:idx<rows.length-1?"1px solid rgba(255,255,255,0.04)":"none",
                  borderLeft:`3px solid ${meta2.dotColor}66`,
                  animation:"fadeUp 0.3s ease both",
                  animationDelay:`${Math.min(idx*30,400)}ms`,
                }}>
                  <div>
                    <div style={{ display:"flex", alignItems:"center", gap:6 }}>
                      <span style={{ fontSize:12, fontWeight:700, color:"#f1f5f9", letterSpacing:"0.04em" }}>{r.item.ticker.replace(".NS","")}</span>
                      <span style={{ fontSize:8, padding:"1px 5px", borderRadius:2, background:"rgba(255,255,255,0.05)", color:"#475569", letterSpacing:"0.12em" }}>{r.sec}</span>
                    </div>
                    <div style={{ fontSize:10, color:"#334155", marginTop:1 }}>{COMPANY[r.item.ticker]??r.item.ticker}</div>
                  </div>
                  <div>{r.pred ? <SignalBadge signal={r.pred.finalSignal} size="sm" /> : <span style={{fontSize:9,color:"#334155"}}>—</span>}</div>
                  <div style={{ textAlign:"right", fontSize:10, color:"#475569" }}>₹{r.entry}</div>
                  <div style={{ textAlign:"right", fontSize:10, color:"#94a3b8" }}>₹{r.cmp}</div>
                  <div style={{ textAlign:"right", fontSize:10, fontWeight:700, color:r.pnlPct>=0?"#10b981":"#f43f5e" }}>
                    {r.pnlPct>=0?"+":""}{r.pnlPct}%
                  </div>
                  <div style={{ textAlign:"right" }}>
                    {r.pred ? (
                      <span style={{ fontSize:10, fontWeight:700, color:meta2.dotColor }}>
                        {Math.round((r.pred.lstmConf??0)*100)}%
                      </span>
                    ) : <span style={{fontSize:9,color:"#334155"}}>—</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Sector allocation + Risk metrics */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
            <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14 }}>
              <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:12 }}>SECTOR ALLOCATION</div>
              {sectorLabels.length>0 ? (
                <>
                  <div style={{ height:160 }}>
                    <Doughnut
                      data={{ labels:sectorLabels, datasets:[{ data:sectorVals, backgroundColor:sectorCols, borderWidth:0 }] }}
                      options={{ responsive:true, maintainAspectRatio:false, cutout:"65%", plugins:{legend:{display:false}} }}
                    />
                  </div>
                  <div style={{ display:"flex", flexWrap:"wrap", gap:"4px 14px", marginTop:10, justifyContent:"center" }}>
                    {sectorLabels.map((s,i)=>(
                      <span key={s} style={{ fontSize:9, color:sectorCols[i], display:"flex", alignItems:"center", gap:4 }}>
                        <span style={{ width:7,height:7,background:sectorCols[i],borderRadius:1,display:"inline-block" }}/>{s} {sectorVals[i]}
                      </span>
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ height:160, display:"flex", alignItems:"center", justifyContent:"center", color:"#334155", fontSize:10 }}>NO DATA</div>
              )}
            </div>

            <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14 }}>
              <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:12 }}>SIGNAL QUALITY METRICS</div>
              {[
                { label:"Avg LSTM Confidence", val: rows.length ? `${Math.round(rows.reduce((s,r)=>s+(r.pred?.lstmConf??0),0)/rows.length*100)}%` : "—", color:"#06b6d4" },
                { label:"Bullish Positions",   val: `${bullRows} / ${rows.length}`,  color:"#10b981" },
                { label:"Bearish Positions",   val: `${rows.filter(r=>r.meta?.direction==="bear").length} / ${rows.length}`, color:"#f43f5e" },
                { label:"Neutral Positions",   val: `${rows.filter(r=>r.meta?.direction==="neutral").length} / ${rows.length}`, color:"#f59e0b" },
                { label:"Positions in Profit", val: `${rows.filter(r=>r.pnlPct>0).length} / ${rows.length}`, color:"#8b5cf6" },
                { label:"Win Rate",            val: `${winRate}%`, color:"#8b5cf6" },
              ].map(m=>(
                <div key={m.label} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"7px 0", borderBottom:"1px solid rgba(255,255,255,0.04)" }}>
                  <span style={{ fontSize:10, color:"#475569" }}>{m.label}</span>
                  <span style={{ fontSize:13, fontWeight:700, color:m.color }}>{m.val}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {isLoading && (
        <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
          {[1,2,3].map(i=>(
            <div key={i} style={{ height:80, background:"#0a0f1a", borderRadius:4, border:"1px solid rgba(255,255,255,0.05)", animation:"skPulse 1.4s ease-in-out infinite", animationDelay:`${i*0.15}s` }} />
          ))}
        </div>
      )}

      <div style={{ marginTop:20, padding:"10px 14px", background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.05)", borderLeft:"2px solid #334155", borderRadius:"0 4px 4px 0", fontSize:9, color:"#1e293b", letterSpacing:"0.08em", lineHeight:1.7,textAlign: "center",}}>
        ENTRY PRICES AND CMP ARE SIMULATED FOR DEMONSTRATION. SIGNAL DATA IS REAL FROM YOUR AI MODEL. NOT FINANCIAL ADVICE.
      </div>

      <style>{`
        @keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
        @keyframes skPulse{0%,100%{opacity:.3}50%{opacity:.7}}
      `}</style>
    </div>
  );
}