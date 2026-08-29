import { useQuery } from '@tanstack/react-query';

import { apiGet } from './client';

export type TraceRecord = {
  trace_id: string;
  route: string;
  model: string;
  plan_steps: number;
  tools: string[];
  retrieved: number;
  reranked: number;
  cache_hit: boolean;
  routing_ms: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  input_tokens: number;
  output_tokens: number;
  confidence: number;
  generated_sql?: string | null;
  events: Array<Record<string, string | number | boolean>>;
};

export function useTracesQuery() {
  return useQuery({
    queryKey: ['observability', 'traces'],
    queryFn: () => apiGet<TraceRecord[]>('/api/observability/traces'),
  });
}
