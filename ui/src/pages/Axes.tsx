import React from "react";
import { api } from "@/lib/api";

export default function Axes() {
  const [names, setNames] = React.useState("autonomy,truth,non_aggression,fairness");
  const [dim, setDim] = React.useState(16);
  const [packId, setPackId] = React.useState<string>("");
  const [active, setActive] = React.useState<any | null>(null);
  const [msg, setMsg] = React.useState<string>("");

  const randomVectors = (n: number, d: number) => {
    return Array.from({ length: n }, () =>
      Array.from({ length: d }, () => +(Math.random() * 2 - 1).toFixed(4))
    );
  };

  const refreshActive = async () => setActive(await api.axes.active());

  React.useEffect(() => { refreshActive(); }, []);

  const build = async () => {
    setMsg("");
    const list = names.split(",").map(s => s.trim()).filter(Boolean);
    if (list.length === 0) return setMsg("Provide at least one axis name.");
    const seed = randomVectors(list.length, dim);
    const res = await api.axes.build(list, seed, { source: "ui" });
    setPackId(res.pack_id);
    await refreshActive();
    setMsg(`Built pack ${res.pack_id} with axes [${res.axes.join(", ")}]`);
  };

  const activate = async () => {
    if (!packId) return setMsg("Enter a pack id to activate.");
    await api.axes.activate(packId);
    await refreshActive();
    setMsg(`Activated ${packId}`);
  };

  return (
    <div>
      <h2>Axes</h2>
      <div style={{ display: "grid", gap: 8, maxWidth: 640 }}>
        <label>Axis names (comma-separated)</label>
        <input value={names} onChange={e => setNames(e.target.value)} />
        <label>Embedding dim (seed vectors)</label>
        <input type="number" value={dim} onChange={e => setDim(+e.target.value)} />
        <button onClick={build}>Build Axis Pack</button>
      </div>
      <div style={{ marginTop: 12 }}>
        <label>Pack ID to activate</label>
        <input value={packId} onChange={e => setPackId(e.target.value)} />
        <button onClick={activate}>Activate</button>
      </div>
      {msg && <div style={{ marginTop: 8, color: "#333" }}>{msg}</div>}
      <h3 style={{ marginTop: 16 }}>Active</h3>
      <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 8 }}>
{JSON.stringify(active, null, 2)}
      </pre>
    </div>
  );
}
