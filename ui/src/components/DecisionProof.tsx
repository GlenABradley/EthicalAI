import React from "react";
export default function DecisionProof({ proof }: { proof: any }) {
  if (!proof) return null;
  const spans = proof.spans ?? [];
  const final = proof.final ?? {};
  return (
    <div style={{ marginTop: 12 }}>
      <h3 style={{ marginBottom: 8 }}>Decision Proof</h3>
      <div style={{ fontSize: 14, color: "#444", marginBottom: 8 }}>
        Action: <b>{final.action}</b> {final.rationale ? `— ${final.rationale}` : ""}
      </div>
      {spans.length > 0 ? (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>i–j</th>
              <th style={{ textAlign: "left" }}>axis</th>
              <th style={{ textAlign: "left" }}>score</th>
              <th style={{ textAlign: "left" }}>τ</th>
            </tr>
          </thead>
          <tbody>
            {spans.map((s: any, idx: number) => (
              <tr key={idx}>
                <td>{s.i}-{s.j}</td>
                <td>{s.axis}</td>
                <td>{s.score?.toFixed?.(3) ?? s.score}</td>
                <td>{s.threshold?.toFixed?.(3) ?? s.threshold}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{ color: "#666" }}>No veto spans.</div>
      )}
    </div>
  );
}
