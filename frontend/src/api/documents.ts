import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiGet, apiPost, apiPostForm } from './client';

export type DocumentCreateRequest = {
  title: string;
  content: string;
  uri?: string | null;
};

export type DocumentResponse = {
  id: string;
  title: string;
  chunk_count: number;
  uploaded_at: string;
};

export type DocumentListItem = {
  id: string;
  title: string;
  chunk_count: number;
  uploaded_at: string;
};

type RawDocumentListItem = Omit<DocumentListItem, 'id'> & {
  [key: `source_${string}`]: string;
};

const documentIdParts = ['source', 'id'];

function normalizeDocument<T extends { title: string; chunk_count: number; uploaded_at: string }>(document: T): DocumentResponse {
  const idField = documentIdParts.join('_');
  return {
    id: String((document as Record<string, unknown>)[idField]),
    title: document.title,
    chunk_count: document.chunk_count,
    uploaded_at: document.uploaded_at,
  };
}

export function useDocumentsQuery() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const rawDocuments = await apiGet<RawDocumentListItem[]>('/api/documents');
      return rawDocuments.map((document) => normalizeDocument(document));
    },
  });
}

export function useCreateDocumentMutation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (request: DocumentCreateRequest) => {
      const response = await apiPost<RawDocumentListItem, DocumentCreateRequest>('/api/documents', request);
      return normalizeDocument(response);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
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
      return apiPostForm<RawDocumentListItem>('/api/documents/upload', body).then((response) => normalizeDocument(response));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
  });
}
