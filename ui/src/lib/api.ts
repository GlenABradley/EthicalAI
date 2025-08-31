const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) }
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  health: () => j<{ status: string; info?: any }>("/health/ready"),
  axes: {
    build: (names: string[], seed_vectors: number[][], meta: Record<string, unknown> = {}) =>
      j<{ pack_id: string; axes: string[] }>("/v1/axes/build", {
        method: "POST",
        body: JSON.stringify({ names, seed_vectors, meta })
      }),
    activate: (pack_id: string) =>
      j<{ ok: boolean }>(`/v1/axes/activate?pack_id=${encodeURIComponent(pack_id)}`, { method: "POST" }),
    active: () => j<{ pack_id: string | null; axes: string[] }>("/v1/axes/active")
  },
  evalText: (text: string, window = 32, stride = 16) =>
    j<{ proof: any; spans: any[]; per_axis?: Record<string, number[]> }>("/v1/eval/text", {
      method: "POST",
      body: JSON.stringify({ text, window, stride })
    }),
  respond: (prompt: string) =>
    j<{ final: string; proof: any; alternatives: Array<{ text: string }> }>("/v1/interaction/respond", {
      method: "POST",
      body: JSON.stringify({ prompt })
    })
};
