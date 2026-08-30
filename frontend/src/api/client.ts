export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export async function apiGet<TResponse>(path: string): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    throw new Error("I couldn't reach RetailData-Pro right now.");
  }

  return response.json() as Promise<TResponse>;
}

export async function apiPost<TResponse, TBody extends object>(path: string, body: TBody): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error("I couldn't complete that request right now.");
  }

  return response.json() as Promise<TResponse>;
}

export async function apiPatch<TResponse, TBody extends object>(path: string, body: TBody): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error('I could not save that change right now.');
  }

  return response.json() as Promise<TResponse>;
}

export async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error('I could not delete that chat right now.');
  }
}

export async function apiPostForm<TResponse>(path: string, body: FormData): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    body,
  });

  if (!response.ok) {
    throw new Error("I couldn't upload that document right now.");
  }

  return response.json() as Promise<TResponse>;
}
