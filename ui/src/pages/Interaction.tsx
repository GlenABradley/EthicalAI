import React from "react";
import { api } from "@/lib/api";
import DecisionProof from "@/components/DecisionProof";

export default function Analyze() {
  const [text, setText] = React.useState("Write something to analyze ethically.");
  const [window, setWindow] = React.useState(16);
  const [stride, setStride] = React.useState(8);
  const [res, setRes] = React.useState<any | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  const run = async () => {
    setErr(null);
    try {
      const r = await api.evalText(text, window, stride);
      setRes(r);
    } catch (e: any) { setErr(e.message); }
  };
  return (
    <div>
      <h2>Analyze</h2>
      <textarea style={{ width: "100%", minHeight: 120 }} value={text} onChange={e => setText(e.target.value)} />
      <div style={{ display: "flex", gap: 12, margin: "8px 0" }}>
        <label>window <input type="number" value={window} onChange={e => setWindow(+e.target.value)} /></label>
        <label>stride <input type="number" value={stride} onChange={e => setStride(+e.target.value)} /></label>
        <button onClick={run}>Evaluate</button>
      </div>
      {err && <div style={{ color: "crimson" }}>{err}</div>}
      {res && (<DecisionProof proof={res.proof} />)}
    </div>
  );
}
