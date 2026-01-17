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

export type TextModelSummary = {
  model_id: string;
  created_at_unix: number;
  checkpoint_path?: string;
  tokenizer_path?: string;
  vocab_size?: number;
  d_model?: number;
  backbone?: string;
  steps?: number;
  device?: string;
};

export type GenerateTextResponse = {
  model_id: string;
  prompt: string;
  text: string;
};

export async function listTextModels(): Promise<TextModelSummary[]> {
  return fetchJson<TextModelSummary[]>('/api/text/models');
}

export async function generateText(payload: {
  prompt: string;
  model_id?: string | null;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
}): Promise<GenerateTextResponse> {
  return fetchJson<GenerateTextResponse>('/api/text/generate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

