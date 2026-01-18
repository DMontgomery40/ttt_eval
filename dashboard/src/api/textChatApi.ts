import { apiUrl } from './config';
import { ChatUpdateEvent } from '../types';

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

export type TextSessionSummary = {
  session_id: string;
  created_at_unix: number;
  updated_at_unix?: number;
  model_id: string;
};

export type CreateSessionRequest = {
  model_id?: string | null;
  lr?: number;
  weight_decay?: number;
  momentum?: number;
  ns_steps?: number;
  steps_per_message?: number;
  chunk_tokens?: number;
};

export async function listTextSessions(limit = 100): Promise<TextSessionSummary[]> {
  return fetchJson<TextSessionSummary[]>(`/api/text/sessions?limit=${encodeURIComponent(String(limit))}`);
}

export async function createTextSession(payload: CreateSessionRequest = {}): Promise<any> {
  return fetchJson('/api/text/sessions', { method: 'POST', body: JSON.stringify(payload) });
}

export async function resetTextSession(sessionId: string): Promise<any> {
  return fetchJson(`/api/text/sessions/${encodeURIComponent(sessionId)}/reset`, { method: 'POST' });
}

export async function chatInSession(payload: {
  session_id: string;
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
}): Promise<{
  session_id: string;
  model_id: string;
  prompt: string;
  completion: string;
  text: string;
  update_events: ChatUpdateEvent[];
  updated_at_unix: number;
}> {
  return fetchJson(`/api/text/sessions/${encodeURIComponent(payload.session_id)}/chat`, {
    method: 'POST',
    body: JSON.stringify({
      prompt: payload.prompt,
      max_new_tokens: payload.max_new_tokens,
      temperature: payload.temperature,
      top_k: payload.top_k,
    }),
  });
}

