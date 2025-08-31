import React from "react";
import { api } from "@/lib/api";

export default function Status() {
  const [loading, setLoading] = React.useState(false);
  const [data, setData] = React.useState<any | null>(null);
  const [active, setActive] = React.useState<any | null>(null);
  const [err, setErr] = React.useState<string | null>(null);
  React.useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [h, a] = await Promise.all([api.health(), api.axes.active()]);
        setData(h); setActive(a);
      } catch (e: any) {
        setErr(e.message);
      } finally { setLoading(false); }
    })();
  }, []);
  return (
    <div>
      <h2>Status</h2>
      {loading && <div>Loading…</div>}
      {err && <div style={{ color: "crimson" }}>{err}</div>}
      {data && (
        <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 8 }}>
{JSON.stringify(data, null, 2)}
        </pre>
      )}
      {active && (
        <>
          <h3 style={{ marginTop: 12 }}>Active Axis Pack</h3>
          <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 8 }}>
{JSON.stringify(active, null, 2)}
          </pre>
        </>
      )}
    </div>
  );
}
