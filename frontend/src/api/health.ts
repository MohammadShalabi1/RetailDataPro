import { useQuery } from '@tanstack/react-query';

import { apiGet } from './client';

export type HealthResponse = {
  status: string;
  service: string;
  database_configured: boolean;
};

export function useHealthQuery() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiGet<HealthResponse>('/api/health'),
    retry: 1,
  });
}
