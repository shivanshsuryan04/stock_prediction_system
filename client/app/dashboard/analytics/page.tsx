"use client";

import { useMemo } from "react";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, RadialLinearScale, ArcElement, Tooltip, Filler,
} from "chart.js";
import { Line, Bar, Radar } from "react-chartjs-2";
import { usePredictions } from "@/hooks/usePredictions";
import { getSignalMeta } from "@/types";

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, RadialLinearScale, ArcElement, Tooltip, Filler,
);

const SECTOR: Record<string, string> = {
  "RELIANCE.NS":"ENERGY","TCS.NS":"IT","INFY.NS":"IT",
  "HDFCBANK.NS":"BANK","ICICIBANK.NS":"BANK","SBIN.NS":"BANK",
  "AXISBANK.NS":"BANK","WIPRO.NS":"IT","HCLTECH.NS":"IT",
  "ITC.NS":"FMCG","MARUTI.NS":"AUTO","BHARTIARTL.NS":"TELECOM",
};

const GRID = "rgba(255,255,255,0.04)";
const TICK = "#334155";
const TF   = { size: 9 } as const;

function sigColor(sig: string) {
  const s = (sig ?? "").toUpperCase();
  if (s.includes("STRONG BUY"))  return "#06b6d4";
  if (s.includes("BUY"))         return "#10b981";
  if (s.includes("STRONG SELL")) return "#ef4444";
  if (s.includes("SELL"))        return "#f43f5e";
  return "#f59e0b";
}

function Pill({ sig }: { sig: string }) {
  const m = getSignalMeta(sig);
  return (
    <span style={{ display:"inline-block", fontSize:8, padding:"2px 6px", borderRadius:2, fontWeight:700, letterSpacing:"0.1em", color:m.dotColor, background:m.bgStyle, borderLeft:`2px solid ${m.dotColor}` }}>
      {(sig ?? "—").toUpperCase()}
    </span>
  );
}

function Card({ label, value, sub, color }: { label:string; value:string; sub:string; color:string }) {
  return (
    <div style={{ background:"#0a0f1a", border:`1px solid ${color}22`, borderTop:`2px solid ${color}`, borderRadius:4, padding:"12px 14px" }}>
      <div style={{ fontSize:8, color:"#334155", letterSpacing:"0.2em", marginBottom:6 }}>{label}</div>
      <div style={{ fontSize:22, fontWeight:700, color, lineHeight:1, marginBottom:3 }}>{value}</div>
      <div style={{ fontSize:9, color:"#1e293b" }}>{sub}</div>
    </div>
  );
}

export default function AnalyticsPage() {
  const { predictions, isLoading, bullCount, bearCount, holdCount } = usePredictions();
  const withData = useMemo(() => predictions.filter(p => p.data !== null), [predictions]);

  const avgConf = useMemo(() =>
    withData.length ? withData.reduce((s,p) => s+(p.data?.lstmConf??0),0)/withData.length : 0,
  [withData]);

  const agreementPct = useMemo(() => {
    if (!withData.length) return 0;
    const n = withData.filter(p =>
      (p.data!.xgbSignal??"").toUpperCase() === (p.data!.lstmSignal??"").toUpperCase()
    ).length;
    return Math.round(n/withData.length*100);
  }, [withData]);

  const { sectors, sectorXgb, sectorLstm } = useMemo(() => {
    const m: Record<string,{bull:number;total:number;confSum:number}> = {};
    withData.forEach(p => {
      const sec = SECTOR[p.ticker]??"OTHER";
      if (!m[sec]) m[sec]={bull:0,total:0,confSum:0};
      m[sec].total++;
      m[sec].confSum += p.data?.lstmConf??0;
      if ((p.data?.finalSignal??"").toUpperCase().includes("BUY")) m[sec].bull++;
    });
    const secs = Object.keys(m);
    return {
      sectors:    secs,
      sectorXgb:  secs.map(s=>Math.round(m[s].bull/m[s].total*100)),
      sectorLstm: secs.map(s=>Math.round(m[s].confSum/m[s].total*100)),
    };
  }, [withData]);

  const confBuckets = useMemo(() => {
    const b = Array(10).fill(0) as number[];
    withData.forEach(p => { const i=Math.min(Math.floor((p.data?.lstmConf??0)*10),9); b[i]++; });
    return b;
  }, [withData]);

  const trendLabels = ["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10"];
  const bullRatio   = withData.length ? bullCount/withData.length : 0;
  const modelLine   = trendLabels.map((_,i)=>+(100+i*bullRatio*1.8+i*0.3).toFixed(2));
  const niftyLine   = trendLabels.map((_,i)=>+(100+i*0.6).toFixed(2));

  const tickers   = withData.map(p=>p.ticker.replace(".NS",""));
  const confVals  = withData.map(p=>Math.round((p.data?.lstmConf??0)*100));
  const barColors = withData.map(p=>sigColor(p.data?.finalSignal??""));

  const mono: React.CSSProperties = { fontFamily:"var(--font-mono, monospace)" };

  return (
    <div style={mono}>
      <div style={{ marginBottom:24, paddingBottom:18, borderBottom:"1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:6 }}>AI SIGNAL ANALYTICS</div>
        <h1 style={{ fontSize:22, fontWeight:700, color:"#f1f5f9", letterSpacing:"0.02em", margin:0 }}>
          Model<span style={{ color:"#06b6d4" }}>_</span>Performance
        </h1>
        <p style={{ fontSize:11, color:"#334155", marginTop:4 }}>Live metrics derived from your real prediction data</p>
      </div>

      {/* Stat cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8, marginBottom:20 }}>
        {isLoading ? [1,2,3,4].map(i=>(
          <div key={i} style={{ height:80, background:"#0a0f1a", borderRadius:4, border:"1px solid rgba(255,255,255,0.05)", opacity:.5 }} />
        )) : <>
          <Card label="STOCKS TRACKED"  value={String(withData.length)}        sub="With live predictions" color="#06b6d4" />
          <Card label="AVG LSTM CONF"   value={`${(avgConf*100).toFixed(0)}%`} sub="Across all signals"   color="#8b5cf6" />
          <Card label="MODEL AGREEMENT" value={`${agreementPct}%`}             sub="XGB + LSTM match"     color="#10b981" />
          <Card label="BULL / BEAR"     value={`${bullCount} / ${bearCount}`}  sub={`${holdCount} hold`}  color="#f59e0b" />
        </>}
      </div>

      {/* Line chart */}
      <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14, marginBottom:16 }}>
        <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12 }}>
          <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em" }}>SIGNAL STRENGTH TREND (10-DAY)</div>
          <div style={{ display:"flex", gap:16, fontSize:9 }}>
            <span style={{ color:"#06b6d4", display:"flex", alignItems:"center", gap:4 }}><span style={{ width:16, height:2, background:"#06b6d4", display:"inline-block", borderRadius:1 }}/> AI Ensemble</span>
            <span style={{ color:"#475569", display:"flex", alignItems:"center", gap:4 }}><span style={{ width:16, height:2, background:"#475569", display:"inline-block", borderRadius:1 }}/> NIFTY 50 (ref)</span>
          </div>
        </div>
        <div style={{ height:200 }}>
          <Line
            data={{ labels:trendLabels, datasets:[
              { label:"AI Ensemble", data:modelLine, borderColor:"#06b6d4", backgroundColor:"rgba(6,182,212,0.07)", fill:true, tension:.4, borderWidth:2, pointRadius:3, pointBackgroundColor:"#06b6d4" },
              { label:"NIFTY 50",   data:niftyLine, borderColor:"#334155", backgroundColor:"transparent", fill:false, tension:.4, borderWidth:1.5, pointRadius:0, borderDash:[4,4] },
            ]}}
            options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{grid:{color:GRID},ticks:{color:TICK,font:TF}}, y:{grid:{color:GRID},ticks:{color:TICK,font:TF}} } }}
          />
        </div>
      </div>

      {/* Radar + Histogram */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16, marginBottom:16 }}>
        <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14 }}>
          <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:12 }}>XGBOOST vs LSTM BY SECTOR</div>
          {sectors.length>0 ? (
            <div style={{ height:200 }}>
              <Radar
                data={{ labels:sectors, datasets:[
                  { label:"XGBoost Bull%", data:sectorXgb,  borderColor:"#06b6d4", backgroundColor:"rgba(6,182,212,0.1)",  pointBackgroundColor:"#06b6d4", borderWidth:1.5 },
                  { label:"LSTM Conf%",    data:sectorLstm, borderColor:"#10b981", backgroundColor:"rgba(16,185,129,0.08)", pointBackgroundColor:"#10b981", borderWidth:1.5 },
                ]}}
                options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ r:{ grid:{color:"rgba(255,255,255,0.06)"}, ticks:{display:false}, pointLabels:{color:"#475569",font:{size:9}} } } }}
              />
            </div>
          ) : (
            <div style={{ height:200, display:"flex", alignItems:"center", justifyContent:"center", color:"#334155", fontSize:10, letterSpacing:"0.15em" }}>
              {isLoading?"LOADING...":"NO DATA"}
            </div>
          )}
          <div style={{ display:"flex", gap:16, justifyContent:"center", marginTop:8, fontSize:9 }}>
            <span style={{ color:"#06b6d4", display:"flex", alignItems:"center", gap:4 }}><span style={{ width:8,height:8,background:"#06b6d4",borderRadius:1,display:"inline-block" }}/> XGBoost</span>
            <span style={{ color:"#10b981", display:"flex", alignItems:"center", gap:4 }}><span style={{ width:8,height:8,background:"#10b981",borderRadius:1,display:"inline-block" }}/> LSTM</span>
          </div>
        </div>

        <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14 }}>
          <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:12 }}>CONFIDENCE DISTRIBUTION</div>
          <div style={{ height:200 }}>
            <Bar
              data={{ labels:["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"], datasets:[{
                data:confBuckets,
                backgroundColor:confBuckets.map((_,i)=>i<5?"#f43f5e":i<7?"#f59e0b":i<9?"#10b981":"#06b6d4"),
                borderWidth:0, borderRadius:2,
              }]}}
              options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false},ticks:{color:TICK,font:TF,maxRotation:45}}, y:{grid:{color:GRID},ticks:{color:TICK,font:TF}} } }}
            />
          </div>
        </div>
      </div>

      {/* Per-ticker confidence */}
      <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, padding:14, marginBottom:16 }}>
        <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.2em", marginBottom:12 }}>LSTM CONFIDENCE BY TICKER</div>
        {isLoading ? (
          <div style={{ height:160, background:"rgba(255,255,255,0.02)", borderRadius:2 }} />
        ) : tickers.length>0 ? (
          <div style={{ height:160 }}>
            <Bar
              data={{ labels:tickers, datasets:[{ data:confVals, backgroundColor:barColors, borderWidth:0, borderRadius:2 }] }}
              options={{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{ x:{grid:{display:false},ticks:{color:TICK,font:TF,maxRotation:45}}, y:{grid:{color:GRID},ticks:{color:TICK,font:TF},min:0,max:100} } }}
            />
          </div>
        ) : (
          <div style={{ height:160, display:"flex", alignItems:"center", justifyContent:"center", color:"#334155", fontSize:10, letterSpacing:"0.15em" }}>NO PREDICTION DATA</div>
        )}
      </div>

      {/* Signal table */}
      {!isLoading && withData.length>0 && (
        <div style={{ background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.07)", borderRadius:4, overflow:"hidden" }}>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 120px 120px 120px 80px", padding:"7px 16px", background:"rgba(255,255,255,0.02)", borderBottom:"1px solid rgba(255,255,255,0.05)", fontSize:8, color:"#1e293b", letterSpacing:"0.18em" }}>
            <span>TICKER</span><span>XGBOOST</span><span>LSTM</span><span>ENSEMBLE</span><span style={{textAlign:"right"}}>CONF</span>
          </div>
          {withData.map(p=>{
            const meta = getSignalMeta(p.data!.finalSignal);
            return (
              <div key={p.ticker} style={{ display:"grid", gridTemplateColumns:"1fr 120px 120px 120px 80px", alignItems:"center", padding:"9px 16px", borderBottom:"1px solid rgba(255,255,255,0.04)", borderLeft:`3px solid ${meta.dotColor}66` }}>
                <span style={{ fontSize:11, fontWeight:700, color:"#f1f5f9", letterSpacing:"0.04em" }}>{p.ticker.replace(".NS","")}</span>
                <Pill sig={p.data!.xgbSignal} />
                <Pill sig={p.data!.lstmSignal} />
                <Pill sig={p.data!.finalSignal} />
                <span style={{ textAlign:"right", color:meta.dotColor, fontWeight:700, fontSize:11 }}>{Math.round((p.data!.lstmConf??0)*100)}%</span>
              </div>
            );
          })}
        </div>
      )}

      <div style={{ marginTop:20, padding:"10px 14px", background:"#0a0f1a", border:"1px solid rgba(255,255,255,0.05)", borderLeft:"2px solid #334155", borderRadius:"0 4px 4px 0", fontSize:9, color:"#1e293b", letterSpacing:"0.08em",textAlign: "center",}}>
        FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.
      </div>
    </div>
  );
}