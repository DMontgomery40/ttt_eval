import { apiUrl } from './config';
import { MonitorEvent } from '../types';

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export type TextRunSummary = {
  run_id: string;
  created_at_unix: number;
  preview?: string;
  summary?: {
    chunks?: number;
    flagged?: number;
    blocked?: number;
    rollbacks?: number;
  };
};

export type TextRunData = {
  run_id: string;
  created_at_unix: number;
  input_text: string;
  summary: {
    chunks: number;
    flagged: number;
    blocked: number;
    rollbacks: number;
  };
  events: MonitorEvent[];
};

export async function listTextRuns(limit = 50): Promise<TextRunSummary[]> {
  return fetchJson<TextRunSummary[]>(`/api/text/runs?limit=${encodeURIComponent(String(limit))}`);
}

export async function getTextRun(runId: string): Promise<TextRunData> {
  return fetchJson<TextRunData>(`/api/text/runs/${encodeURIComponent(runId)}`);
}

export async function createTextRun(payload: Record<string, any>): Promise<TextRunData> {
  return fetchJson<TextRunData>('/api/text/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

