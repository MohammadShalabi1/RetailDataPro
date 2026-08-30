import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiDelete, apiGet, apiPatch, apiPost } from './client';

export type ClientCitation = {
  label: string;
  claim?: string | null;
  excerpt?: string | null;
};

export type ClientMessage = {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  citations: ClientCitation[];
  status: 'complete' | 'failed';
};

export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_at?: string | null;
};

export type ConversationDetail = ConversationSummary & {
  messages: ClientMessage[];
};

export type SendMessageRequest = {
  message: string;
};

export type SendMessageResponse = {
  conversation_id: string;
  message: ClientMessage;
};

export function useConversationsQuery() {
  return useQuery({
    queryKey: ['conversations'],
    queryFn: () => apiGet<ConversationSummary[]>('/api/conversations'),
  });
}

export function useConversationQuery(conversationId?: string) {
  return useQuery({
    queryKey: ['conversations', conversationId],
    queryFn: () => apiGet<ConversationDetail>(`/api/conversations/${conversationId}`),
    enabled: Boolean(conversationId),
  });
}

export function useCreateConversationMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiPost<ConversationSummary, Record<string, never>>('/api/conversations', {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });
}

export function useSendMessageMutation(conversationId?: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: SendMessageRequest) =>
      apiPost<SendMessageResponse, SendMessageRequest>(`/api/conversations/${conversationId}/messages`, request),
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['conversations', response.conversation_id] });
    },
  });
}

export function useRenameConversationMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ conversationId, title }: { conversationId: string; title: string }) =>
      apiPatch<ConversationSummary, { title: string }>(`/api/conversations/${conversationId}`, { title }),
    onSuccess: (conversation) => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
      queryClient.invalidateQueries({ queryKey: ['conversations', conversation.id] });
    },
  });
}

export function useDeleteConversationMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (conversationId: string) => apiDelete(`/api/conversations/${conversationId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conversations'] });
    },
  });
}
