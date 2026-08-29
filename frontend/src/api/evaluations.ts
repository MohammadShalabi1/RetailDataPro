import { useQuery } from '@tanstack/react-query';

import { apiGet } from './client';

export type EvalMetric = {
  name: string;
  value: number;
  unit: string;
};

export type EvalRun = {
  run_id: string;
  metrics: EvalMetric[];
  datasets: string[];
  rows_evaluated: number;
  breakdown: Record<string, number>;
};

export function useLatestEvaluationQuery() {
  return useQuery({
    queryKey: ['evaluations', 'latest'],
    queryFn: () => apiGet<EvalRun>('/api/evaluations/latest'),
  });
}
