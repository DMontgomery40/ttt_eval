export const API_BASE_URL = (import.meta.env.VITE_NANO_API_URL || '').trim();

export function apiUrl(path: string): string {
  if (!path.startsWith('/')) return apiUrl(`/${path}`);
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

