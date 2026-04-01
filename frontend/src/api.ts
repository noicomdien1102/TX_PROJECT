// ─── Types ────────────────────────────────────────────────────────────────────

export type Side = 'T' | 'X';

export interface ManualEntry {
  source: 'dice' | 'manual' | 'unknown';
  dice: number[] | null;
  sum: number | null;
  result: Side;
}

export interface PredictResponse {
  total_entries: number;
  sequence_tail: Side[];
  recent_manual: ManualEntry[];
  probabilities: { T: number; X: number };
  suggest: Side;
  patterns: { name: string; detail: string; prob: number; percentage_str: string; expected: Side }[];
  bias: {
    has_bias: boolean;
    biased_faces: number[];
    face_pcts: Record<string, number>;
  };
}

export interface UploadImageResponse {
  count: number;
  results: Side[];
  message: string;
  debug_log: any[];
}

export interface AppendDiceResponse {
  dice: number[];
  sum: number;
  result: Side;
  total_entries: number;
}

// ─── API Client ───────────────────────────────────────────────────────────────

// If injected by Render, use it (adds https://). Otherwise fallback to local Vite proxy (/api)
const BASE = import.meta.env.VITE_API_URL 
  ? `https://${import.meta.env.VITE_API_URL}` 
  : '/api';

export const api = {
  async predict(): Promise<PredictResponse> {
    const res = await fetch(`${BASE}/predict`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async appendDice(d1: number, d2: number, d3: number): Promise<AppendDiceResponse> {
    const res = await fetch(`${BASE}/append-dice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ d1, d2, d3 }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async appendManual(result: Side): Promise<{result: Side, total_entries: number}> {
    const res = await fetch(`${BASE}/append-manual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async uploadImage(file: File): Promise<UploadImageResponse> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/upload-image`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },

  async reset(): Promise<void> {
    const res = await fetch(`${BASE}/reset`, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
  },
};
