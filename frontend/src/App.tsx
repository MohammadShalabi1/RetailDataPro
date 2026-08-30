import { ChangeEvent, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  FileText,
  Loader2,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';
import { Link, useNavigate, useParams } from 'react-router-dom';

import {
  ClientMessage,
  ConversationSummary,
  useConversationQuery,
  useConversationsQuery,
  useCreateConversationMutation,
  useDeleteConversationMutation,
  useRenameConversationMutation,
  useSendMessageMutation,
} from './api/conversations';
import { useCreateDocumentMutation, useDocumentsQuery, useUploadDocumentMutation } from './api/documents';
import { useHealthQuery } from './api/health';

type PendingUserMessage = {
  id: string;
  role: 'user';
  content: string;
  created_at: string;
  citations: [];
  status: 'complete';
};

const EXAMPLE_PROMPTS = [
  'Which suppliers had the weakest fulfillment reliability?',
  'Which categories are losing revenue?',
  'What does the supplier report say about inventory risk?',
  'Compare database sales with uploaded supplier documentation.',
];

export function App() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const conversationsQuery = useConversationsQuery();
  const conversationQuery = useConversationQuery(conversationId);
  const healthQuery = useHealthQuery();
  const createConversation = useCreateConversationMutation();
  const deleteConversation = useDeleteConversationMutation();
  const [isDocumentsOpen, setIsDocumentsOpen] = useState(false);

  async function startConversation() {
    const conversation = await createConversation.mutateAsync();
    navigate(`/chat/${conversation.id}`);
  }

  async function deleteActiveConversation(id: string) {
    const confirmed = window.confirm('Delete this chat? This cannot be undone.');
    if (!confirmed) {
      return;
    }
    await deleteConversation.mutateAsync(id);
    navigate('/');
  }

  return (
    <main className="app-shell">
      <Sidebar
        activeConversationId={conversationId}
        conversations={conversationsQuery.data ?? []}
        isLoading={conversationsQuery.isLoading}
        isCreating={createConversation.isPending}
        onNewChat={startConversation}
        onOpenDocuments={() => setIsDocumentsOpen((value) => !value)}
      />
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>RetailData-Pro AI Assistant</h1>
            <p>Ask about sales, suppliers, inventory, or uploaded reports.</p>
          </div>
          <div className="service-status" data-state={healthQuery.data?.status === 'ok' ? 'ok' : 'checking'}>
            <span />
            {healthQuery.isLoading ? 'Checking connection' : healthQuery.data?.status === 'ok' ? 'Operational' : 'Unavailable'}
          </div>
        </header>

        <div className="content-grid" data-documents-open={isDocumentsOpen}>
          <ChatPage
            conversationId={conversationId}
            conversation={conversationQuery.data}
            isLoading={conversationQuery.isLoading}
            isError={conversationQuery.isError}
            onCreateConversation={startConversation}
            onDeleteConversation={deleteActiveConversation}
            onOpenDocuments={() => setIsDocumentsOpen(true)}
          />
          {isDocumentsOpen ? <DocumentPanel /> : null}
        </div>
      </section>
    </main>
  );
}

function Sidebar({
  activeConversationId,
  conversations,
  isLoading,
  isCreating,
  onNewChat,
  onOpenDocuments,
}: {
  activeConversationId?: string;
  conversations: ConversationSummary[];
  isLoading: boolean;
  isCreating: boolean;
  onNewChat: () => void;
  onOpenDocuments: () => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <strong>RetailData-Pro</strong>
          <small>Private retail assistant</small>
        </div>
      </div>

      <button className="new-chat-button" type="button" onClick={onNewChat} disabled={isCreating}>
        {isCreating ? <Loader2 size={18} aria-hidden="true" /> : <Plus size={18} aria-hidden="true" />}
        New Chat
      </button>

      <button className="sidebar-tool-button" type="button" onClick={onOpenDocuments}>
        <Upload size={17} aria-hidden="true" />
        Documents
      </button>

      <div className="history-header">
        <MessageSquare size={16} aria-hidden="true" />
        <span>Chat History</span>
      </div>

      <nav className="history-list" aria-label="Chat history">
        {isLoading ? <p className="sidebar-note">Loading chats...</p> : null}
        {!isLoading && conversations.length === 0 ? <p className="sidebar-note">No chats yet.</p> : null}
        {conversations.map((conversation) => (
          <Link
            className={conversation.id === activeConversationId ? 'history-item active' : 'history-item'}
            to={`/chat/${conversation.id}`}
            key={conversation.id}
          >
            <span>{conversation.title}</span>
            <time>{formatRelativeDate(conversation.last_message_at ?? conversation.updated_at)}</time>
          </Link>
        ))}
      </nav>
    </aside>
  );
}

function ChatPage({
  conversationId,
  conversation,
  isLoading,
  isError,
  onCreateConversation,
  onDeleteConversation,
  onOpenDocuments,
}: {
  conversationId?: string;
  conversation?: { title: string; messages: ClientMessage[] };
  isLoading: boolean;
  isError: boolean;
  onCreateConversation: () => void;
  onDeleteConversation: (conversationId: string) => void;
  onOpenDocuments: () => void;
}) {
  const [draft, setDraft] = useState('');
  const [pendingMessage, setPendingMessage] = useState<PendingUserMessage | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const sendMessage = useSendMessageMutation(conversationId);
  const renameConversation = useRenameConversationMutation();

  const messages = useMemo(() => {
    const saved = conversation?.messages ?? [];
    return pendingMessage ? [...saved, pendingMessage] : saved;
  }, [conversation?.messages, pendingMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, sendMessage.isPending]);

  useEffect(() => {
    if (conversation?.title) {
      setTitleDraft(conversation.title);
    }
  }, [conversation?.title]);

  async function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = draft.trim();
    if (!conversationId || !message || sendMessage.isPending) {
      return;
    }

    setSendError(null);
    setDraft('');
    setPendingMessage({
      id: `pending-${Date.now()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
      citations: [],
      status: 'complete',
    });

    try {
      await sendMessage.mutateAsync({ message });
      setPendingMessage(null);
    } catch (error) {
      setSendError(error instanceof Error ? error.message : "I couldn't complete that request right now.");
      setPendingMessage(null);
    }
  }

  async function saveTitle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!conversationId || !titleDraft.trim()) {
      return;
    }
    await renameConversation.mutateAsync({ conversationId, title: titleDraft.trim() });
    setRenaming(false);
  }

  if (!conversationId) {
    return <EmptyStart onCreateConversation={onCreateConversation} />;
  }

  if (isLoading) {
    return <div className="chat-surface centered-state">Loading conversation...</div>;
  }

  if (isError || !conversation) {
    return <div className="chat-surface centered-state">I couldn't load that chat. Please choose another conversation.</div>;
  }

  return (
    <section className="chat-surface">
      <div className="chat-title-row">
        {renaming ? (
          <form className="rename-form" onSubmit={saveTitle}>
            <input aria-label="Conversation title" value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} />
            <button type="submit" disabled={renameConversation.isPending}>
              Save
            </button>
          </form>
        ) : (
          <div>
            <h2>{conversation.title}</h2>
            <p>{messages.length === 0 ? 'New conversation' : `${messages.length} messages`}</p>
          </div>
        )}
        <div className="chat-actions">
          <button type="button" onClick={() => setRenaming((value) => !value)} aria-label="Rename chat">
            <Pencil size={17} aria-hidden="true" />
          </button>
          <button type="button" onClick={() => onDeleteConversation(conversationId)} aria-label="Delete chat">
            <Trash2 size={17} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="message-list" aria-live="polite">
        {messages.length === 0 ? <PromptSuggestions onPromptClick={setDraft} /> : null}
        {messages.map((message) => (
          <MessageBubble message={message} key={message.id} />
        ))}
        {sendMessage.isPending ? <AssistantLoading /> : null}
        <div ref={messagesEndRef} />
      </div>

      {sendError ? (
        <div className="error-banner">
          <AlertCircle size={17} aria-hidden="true" />
          {sendError}
        </div>
      ) : null}

      <form className="composer" onSubmit={submitMessage}>
        <button type="button" aria-label="Open documents" onClick={onOpenDocuments}>
          <Upload size={18} aria-hidden="true" />
        </button>
        <input
          aria-label="Message RetailData-Pro AI Assistant"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Message RetailData-Pro AI Assistant..."
        />
        <button type="submit" disabled={sendMessage.isPending || draft.trim().length === 0} aria-label="Send message">
          {sendMessage.isPending ? <Loader2 size={18} aria-hidden="true" /> : <Send size={18} aria-hidden="true" />}
        </button>
      </form>
    </section>
  );
}

function EmptyStart({ onCreateConversation }: { onCreateConversation: () => void }) {
  return (
    <section className="chat-surface empty-chat">
      <Sparkles size={28} aria-hidden="true" />
      <h2>Ask about sales, suppliers, inventory, or uploaded reports.</h2>
      <p>Select a previous chat from the sidebar or start a new one.</p>
      <button type="button" onClick={onCreateConversation}>
        <Plus size={18} aria-hidden="true" />
        New Chat
      </button>
      <PromptSuggestions />
    </section>
  );
}

function PromptSuggestions({ onPromptClick }: { onPromptClick?: (prompt: string) => void }) {
  return (
    <div className="prompt-grid">
      {EXAMPLE_PROMPTS.map((prompt) => (
        <button type="button" onClick={() => onPromptClick?.(prompt)} key={prompt}>
          {prompt}
        </button>
      ))}
    </div>
  );
}

function MessageBubble({ message }: { message: ClientMessage | PendingUserMessage }) {
  return (
    <article className={`message ${message.role} ${message.status === 'failed' ? 'failed' : ''}`}>
      <div className="message-avatar">{message.role === 'user' ? 'You' : <Sparkles size={16} aria-hidden="true" />}</div>
      <div className="message-body">
        <p>{message.content}</p>
        {'citations' in message && message.citations.length > 0 ? (
          <div className="citation-list" aria-label="Citations">
            {message.citations.map((citation) => (
              <div className="citation-card" key={`${citation.label}-${citation.claim ?? ''}`}>
                <FileText size={16} aria-hidden="true" />
                <div>
                  <strong>{citation.label}</strong>
                  {citation.claim ? <span>{citation.claim}</span> : null}
                  {citation.excerpt ? <p>{citation.excerpt}</p> : null}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function AssistantLoading() {
  return (
    <article className="message assistant loading">
      <div className="message-avatar">
        <Sparkles size={16} aria-hidden="true" />
      </div>
      <div className="message-body typing">
        <span>RetailData-Pro is preparing an answer</span>
        <i />
        <i />
        <i />
      </div>
    </article>
  );
}

function DocumentPanel() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [fileError, setFileError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const documentsQuery = useDocumentsQuery();
  const createDocument = useCreateDocumentMutation();
  const uploadDocument = useUploadDocumentMutation();
  const isUploading = createDocument.isPending || uploadDocument.isPending;

  async function uploadFile(file: File) {
    setFileError(null);
    const allowedExtensions = ['.txt', '.md', '.csv', '.json', '.pdf'];
    const lowerName = file.name.toLowerCase();
    const isAllowed = allowedExtensions.some((extension) => lowerName.endsWith(extension)) || file.type.startsWith('text/');
    if (!isAllowed) {
      setFileError('Use a PDF, TXT, Markdown, CSV, or JSON file.');
      return;
    }
    if (file.size > 10_000_000) {
      setFileError('Use a file under 10 MB.');
      return;
    }

    const uploadTitle = file.name.replace(/\.[^/.]+$/, '');
    try {
      await uploadDocument.mutateAsync({ title: uploadTitle, file });
    } catch (error) {
      setFileError(error instanceof Error ? error.message : "I couldn't upload that document right now.");
    }
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      void uploadFile(file);
    }
    event.target.value = '';
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) {
      void uploadFile(file);
    }
  }

  async function submitDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedTitle = title.trim();
    const trimmedContent = content.trim();
    if (!trimmedTitle || !trimmedContent || isUploading) {
      return;
    }
    try {
      await createDocument.mutateAsync({ title: trimmedTitle, content: trimmedContent });
      setTitle('');
      setContent('');
      setFileError(null);
    } catch (error) {
      setFileError(error instanceof Error ? error.message : "I couldn't add that document right now.");
    }
  }

  return (
    <aside className="documents-panel">
      <div className="panel-heading">
        <div>
          <h2>Documents</h2>
          <p>Uploaded reports are available to this assistant.</p>
        </div>
        <Search size={18} aria-hidden="true" />
      </div>

      <form className="document-form" onSubmit={submitDocument}>
        <div
          className={`drop-zone${isDragging ? ' dragging' : ''}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <Upload size={18} aria-hidden="true" />
          <span>{uploadDocument.isPending ? 'Uploading document...' : 'Drop a PDF or text report'}</span>
          <button type="button" onClick={() => fileInputRef.current?.click()}>
            Choose
          </button>
          <input
            ref={fileInputRef}
            aria-label="Upload document"
            type="file"
            accept=".pdf,.txt,.md,.csv,.json,text/*,application/pdf"
            onChange={handleFileInput}
          />
        </div>

        <input aria-label="Document title" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Report title" />
        <textarea
          aria-label="Document content"
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Paste report text..."
        />
        <button type="submit" disabled={isUploading || !title.trim() || !content.trim()}>
          {createDocument.isPending ? 'Adding document' : 'Add pasted text'}
        </button>
        {fileError ? <p className="form-error">{fileError}</p> : null}
      </form>

      <div className="document-list">
        {documentsQuery.isLoading ? <p>Loading documents...</p> : null}
        {!documentsQuery.isLoading && (documentsQuery.data ?? []).length === 0 ? <p>No documents uploaded yet.</p> : null}
        {(documentsQuery.data ?? []).map((document) => (
          <div className="document-row" key={document.id}>
            <FileText size={18} aria-hidden="true" />
            <div>
              <strong>{document.title}</strong>
              <span>
                {document.chunk_count} sections - {formatRelativeDate(document.uploaded_at)}
              </span>
            </div>
            <MoreHorizontal size={17} aria-hidden="true" />
          </div>
        ))}
      </div>
    </aside>
  );
}

function formatRelativeDate(value: string) {
  const date = new Date(value);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const dayMs = 24 * 60 * 60 * 1000;
  if (diffMs < dayMs) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }
  if (diffMs < dayMs * 7) {
    return date.toLocaleDateString([], { weekday: 'short' });
  }
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}
