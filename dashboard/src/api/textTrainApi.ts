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

export type TrainJobSummary = {
  model_id: string;
  pid: number;
  started_at_unix: number;
  status: string;
  exit_code: number | null;
};

export type TrainStatus = {
  model_id: string;
  status: string;
  pid: number | null;
  started_at_unix: number | null;
  exit_code: number | null;
  error?: string | null;
  latest: {
    step?: number;
    loss?: number;
    grad_norm?: number;
    seconds?: number;
    tokens?: number;
  } | null;
};

export type TrainMetric = {
  step: number;
  loss: number;
  grad_norm?: number;
  seconds?: number;
  tokens?: number;
};

export type StartTrainRequest = {
  corpus_paths: string[];
  tokenizer_path?: string | null;
  vocab_size: number;
  d_model: number;
  backbone: 'gru' | 'ssm';
  seq_len: number;
  batch_size: number;
  steps: number;
  seed: number;
  device: 'auto' | 'cpu' | 'mps';
  lr: number;
  weight_decay: number;
  momentum: number;
  ns_steps: number;
  log_every: number;
  save_every: number;
};

export async function startTraining(payload: StartTrainRequest): Promise<{ model_id: string; pid: number }> {
  return fetchJson<{ model_id: string; pid: number }>('/api/text/train', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function listTrainJobs(): Promise<TrainJobSummary[]> {
  return fetchJson<TrainJobSummary[]>('/api/text/train/jobs');
}

export async function getTrainStatus(modelId: string): Promise<TrainStatus> {
  return fetchJson<TrainStatus>(`/api/text/train/${encodeURIComponent(modelId)}/status`);
}

export async function getTrainMetrics(modelId: string, limit = 500): Promise<TrainMetric[]> {
  return fetchJson<TrainMetric[]>(
    `/api/text/train/${encodeURIComponent(modelId)}/metrics?limit=${encodeURIComponent(String(limit))}`
  );
}

export async function cancelTraining(modelId: string): Promise<any> {
  return fetchJson(`/api/text/train/${encodeURIComponent(modelId)}/cancel`, { method: 'POST' });
}
