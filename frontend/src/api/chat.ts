import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiPost } from './client';

export type ChatRequest = {
  question: string;
  conversation_id?: string | null;
  document_source_ids?: string[];
};

export type ChatResponse = {
  answer: string;
  trace_id: string;
  route: string | null;
  confidence: number;
  limitations: string[];
  model: string | null;
  tool_results: Array<Record<string, unknown>>;
};

export function useChatMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: ChatRequest) => apiPost<ChatResponse, ChatRequest>('/api/chat', request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observability', 'traces'] });
    },
  });
}
