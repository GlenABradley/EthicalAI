import React, { useState } from "react";
import Status from "./pages/Status";
import Axes from "./pages/Axes";
import Analyze from "./pages/Analyze";
import Interaction from "./pages/Interaction";

type Tab = "status" | "axes" | "analyze" | "interaction";

export default function App() {
  const [tab, setTab] = useState<Tab>("status");
  return (
    <div style={{ fontFamily: "ui-sans-serif, system-ui", padding: 16, maxWidth: 960, margin: "0 auto" }}>
      <header style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ fontSize: 22, marginRight: 12 }}>EthicalAI</h1>
        {(["status", "axes", "analyze", "interaction"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: "6px 10px",
              borderRadius: 8,
              border: "1px solid #ddd",
              background: tab === t ? "#eef" : "#fff",
              cursor: "pointer"
            }}
          >
            {t}
          </button>
        ))}
        <div style={{ marginLeft: "auto", color: "#666" }}>
          API: {import.meta.env.VITE_API_BASE || "http://localhost:8000"}
        </div>
      </header>
      <main style={{ border: "1px solid #eee", borderRadius: 12, padding: 16 }}>
        {tab === "status" && <Status />}
        {tab === "axes" && <Axes />}
        {tab === "analyze" && <Analyze />}
        {tab === "interaction" && <Interaction />}
      </main>
    </div>
  );
}
