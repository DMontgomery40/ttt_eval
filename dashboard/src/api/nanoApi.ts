import type { SessionData, SessionIndex } from '../types';
import { apiUrl } from './config';

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

export async function fetchSessionIndex(): Promise<SessionIndex> {
  return fetchJson<SessionIndex>('/api/index');
}

export async function fetchSessions(): Promise<SessionData[]> {
  return fetchJson<SessionData[]>('/api/sessions');
}

export async function forkSession(
  parentSessionId: string,
  childSessionId: string,
  options?: { copyOptimizer?: boolean; resetOptimizer?: boolean }
): Promise<SessionData> {
  return fetchJson<SessionData>(`/api/sessions/${encodeURIComponent(parentSessionId)}/fork`, {
    method: 'POST',
    body: JSON.stringify({
      child_session_id: childSessionId,
      copy_optimizer: options?.copyOptimizer ?? true,
      reset_optimizer: options?.resetOptimizer ?? false,
    }),
  });
}

