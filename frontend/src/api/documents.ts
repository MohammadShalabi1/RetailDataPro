import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiPost, apiPostForm } from './client';

export type DocumentCreateRequest = {
  title: string;
  content: string;
  uri?: string | null;
};

export type DocumentResponse = {
  source_id: string;
  title: string;
  chunk_count: number;
  uploaded_at: string;
};

export function useCreateDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: DocumentCreateRequest) => apiPost<DocumentResponse, DocumentCreateRequest>('/api/documents', request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observability', 'traces'] });
    },
  });
}

export function useUploadDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ title, file }: { title: string; file: File }) => {
      const body = new FormData();
      body.append('title', title);
      body.append('file', file);
      return apiPostForm<DocumentResponse>('/api/documents/upload', body);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['observability', 'traces'] });
    },
  });
}
