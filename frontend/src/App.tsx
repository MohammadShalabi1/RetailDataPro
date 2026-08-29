import { ChangeEvent, DragEvent, FormEvent, useRef, useState } from 'react';
import {
  Activity,
  BarChart3,
  Boxes,
  ClipboardList,
  Database,
  FileSearch,
  Gauge,
  Home,
  MessageSquare,
  Upload,
  Search,
  Send,
  ShieldCheck,
  Timer,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Link } from 'react-router-dom';

import { ChatResponse, useChatMutation } from './api/chat';
import { useCreateDocumentMutation, useUploadDocumentMutation } from './api/documents';
import { useLatestEvaluationQuery } from './api/evaluations';
import { useHealthQuery } from './api/health';
import { useTracesQuery } from './api/observability';

type AppProps = {
  view?: 'overview' | 'traces' | 'evaluations';
};

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  response?: ChatResponse;
};

const pieColors = ['#0f8b84', '#f2b84b', '#d65a50', '#96a3ad'];

export function App({ view = 'overview' }: AppProps) {
  const healthQuery = useHealthQuery();

  return (
    <main className="admin-shell">
      <Sidebar active={view} />
      <section className="admin-main">
        <header className="admin-header">
          <div>
            <h1>{view === 'evaluations' ? 'Evaluation Dashboard' : view === 'traces' ? 'AI Trace Viewer' : 'RetailData-Pro Chat'}</h1>
            <p>Ask the AI retail intelligence agent and inspect its operational trace.</p>
          </div>
          <div className="header-actions">
            <div className="search-box">
              <Search size={16} aria-hidden="true" />
              <span>Search traces, queries, sessions...</span>
            </div>
            <div className="api-status" data-state={healthQuery.data?.status ?? 'loading'}>
              <Activity size={16} aria-hidden="true" />
              <span>{healthQuery.isLoading ? 'Checking API' : healthQuery.data?.status ?? 'Unavailable'}</span>
            </div>
          </div>
        </header>

        {view === 'evaluations' ? <EvaluationDashboard /> : view === 'traces' ? <TraceViewer /> : <ChatWorkspace />}
      </section>
    </main>
  );
}

function Sidebar({ active }: { active: AppProps['view'] }) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <Database size={26} aria-hidden="true" />
        <div>
          <strong>RetailData-Pro</strong>
          <span>AI Retail Intelligence</span>
        </div>
      </div>
      <nav aria-label="Admin navigation">
        <Link className={active === 'overview' ? 'active' : ''} to="/">
          <Home size={18} aria-hidden="true" />
          Chat
        </Link>
        <Link className={active === 'traces' ? 'active' : ''} to="/admin/traces">
          <FileSearch size={18} aria-hidden="true" />
          AI Trace Viewer
        </Link>
        <Link className={active === 'evaluations' ? 'active' : ''} to="/admin/evaluations">
          <BarChart3 size={18} aria-hidden="true" />
          Evaluation Dashboard
        </Link>
      </nav>
      <div className="sidebar-foot">No provider keys or hidden chain-of-thought are exposed.</div>
    </aside>
  );
}

function ChatWorkspace() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeDocumentSourceIds, setActiveDocumentSourceIds] = useState<string[]>([]);
  const chatMutation = useChatMutation();

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || chatMutation.isPending) {
      return;
    }

    setQuestion('');
    setMessages((current) => [...current, { role: 'user', content: trimmed }]);
    try {
      const response = await chatMutation.mutateAsync({
        question: trimmed,
        document_source_ids: activeDocumentSourceIds,
      });
      setMessages((current) => [...current, { role: 'assistant', content: response.answer, response }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: error instanceof Error ? error.message : 'The chat request failed.',
        },
      ]);
    }
  }

  return (
    <section className="chat-layout">
      <div className="chat-column">
        <DocumentPanel onDocumentReady={(sourceId) => setActiveDocumentSourceIds([sourceId])} />
        <div className="panel chat-panel">
          <div className="chat-heading">
            <MessageSquare size={20} aria-hidden="true" />
            <div>
              <h2>Retail intelligence chat</h2>
              <p>Responses come from the backend orchestration endpoint.</p>
            </div>
          </div>
          <div className="message-list" aria-live="polite">
            {messages.length === 0 ? (
              <div className="empty-state">
                Ask a question to create a real chat response and trace record.
              </div>
            ) : (
              messages.map((message, index) => (
                <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <span>{message.role}</span>
                  <p>{message.content}</p>
                  {message.response ? <ChatMetadata response={message.response} /> : null}
                </article>
              ))
            )}
          </div>
          <form className="chat-form" onSubmit={submitQuestion}>
            <input
              aria-label="Ask RetailData-Pro"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about revenue, inventory, products, suppliers, or reports..."
            />
            <button type="submit" disabled={chatMutation.isPending || question.trim().length === 0}>
              <Send size={17} aria-hidden="true" />
              <span>{chatMutation.isPending ? 'Sending' : 'Send'}</span>
            </button>
          </form>
        </div>
      </div>
      <TraceViewer compact />
    </section>
  );
}

function DocumentPanel({ onDocumentReady }: { onDocumentReady: (sourceId: string) => void }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [lastDocument, setLastDocument] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const createDocument = useCreateDocumentMutation();
  const uploadDocument = useUploadDocumentMutation();
  const isUploading = createDocument.isPending || uploadDocument.isPending;

  async function uploadFile(file: File) {
    setFileError(null);
    const allowedExtensions = ['.txt', '.md', '.csv', '.json', '.pdf'];
    const lowerName = file.name.toLowerCase();
    const isAllowed = allowedExtensions.some((extension) => lowerName.endsWith(extension)) || file.type.startsWith('text/');
    if (!isAllowed) {
      setFileError('Use a PDF or text file such as .txt, .md, .csv, or .json.');
      return;
    }
    if (file.size > 10_000_000) {
      setFileError('Use a file under 10 MB.');
      return;
    }

    const uploadTitle = file.name.replace(/\.[^/.]+$/, '');
    setTitle(uploadTitle);
    try {
      const response = await uploadDocument.mutateAsync({ title: uploadTitle, file });
      setLastDocument(`${response.title} (${response.chunk_count} chunks)`);
      onDocumentReady(response.source_id);
      setTitle('');
      setContent('');
    } catch (error) {
      setFileError(error instanceof Error ? error.message : 'Document upload failed.');
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
    const response = await createDocument.mutateAsync({ title: trimmedTitle, content: trimmedContent });
    setLastDocument(`${response.title} (${response.chunk_count} chunks)`);
    onDocumentReady(response.source_id);
    setTitle('');
    setContent('');
  }

  return (
    <form className="panel document-panel" onSubmit={submitDocument}>
      <div className="chat-heading">
        <Upload size={20} aria-hidden="true" />
        <div>
          <h2>Add document evidence</h2>
          <p>Paste report text here before asking document questions.</p>
        </div>
      </div>
      <div className="document-grid">
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
          <span>{uploadDocument.isPending ? 'Uploading and extracting document...' : 'Drop a PDF or text report to upload'}</span>
          <button type="button" onClick={() => fileInputRef.current?.click()}>
            Choose file
          </button>
          <input
            ref={fileInputRef}
            aria-label="Upload document file"
            type="file"
            accept=".pdf,.txt,.md,.csv,.json,text/*,application/pdf"
            onChange={handleFileInput}
          />
        </div>
        <input
          aria-label="Document title"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Supplier report title"
        />
        <textarea
          aria-label="Document content"
          value={content}
          onChange={(event) => setContent(event.target.value)}
          placeholder="Paste report text, or choose a PDF/text file..."
        />
      </div>
      <div className="document-actions">
        <button type="submit" disabled={isUploading || !title.trim() || !content.trim()}>
          <Upload size={16} aria-hidden="true" />
          <span>{createDocument.isPending ? 'Adding' : 'Add pasted text'}</span>
        </button>
        {lastDocument ? <span>{lastDocument}</span> : null}
        {fileError ? <span>{fileError}</span> : null}
        {createDocument.isError || uploadDocument.isError ? <span>Document upload failed.</span> : null}
      </div>
    </form>
  );
}

function ChatMetadata({ response }: { response: ChatResponse }) {
  return (
    <dl className="chat-meta">
      <div>
        <dt>Trace</dt>
        <dd>{response.trace_id}</dd>
      </div>
      <div>
        <dt>Route</dt>
        <dd>{response.route ?? 'blocked'}</dd>
      </div>
      <div>
        <dt>Model</dt>
        <dd>{response.model ?? 'none'}</dd>
      </div>
      <div>
        <dt>Confidence</dt>
        <dd>{response.confidence.toFixed(2)}</dd>
      </div>
    </dl>
  );
}

function TraceViewer({ compact = false }: { compact?: boolean }) {
  const tracesQuery = useTracesQuery();
  const traces = tracesQuery.data ?? [];
  const trace = traces[0];

  if (tracesQuery.isLoading) {
    return <div className="panel">Loading trace metadata...</div>;
  }

  if (!trace) {
    return <div className="panel empty-state">No trace records yet. Send a chat message to generate one.</div>;
  }

  const metricCards = [
    { label: 'Confidence', value: trace.confidence.toFixed(2), icon: Gauge },
    { label: 'Latency', value: `${trace.total_ms}ms`, icon: Timer },
    { label: 'Tokens', value: `${trace.input_tokens + trace.output_tokens}`, icon: ClipboardList },
    { label: 'Tool Calls', value: `${trace.tools.length}`, icon: Boxes },
  ];

  return (
    <div className="dashboard-stack">
      <section className="trace-summary panel">
        <div>
          <span className="label">Trace ID</span>
          <strong>{trace.trace_id}</strong>
        </div>
        <div>
          <span className="label">Route</span>
          <strong>{trace.route}</strong>
        </div>
        <div>
          <span className="label">Model</span>
          <strong>{trace.model}</strong>
        </div>
        <div>
          <span className="label">Cache</span>
          <strong>{trace.cache_hit ? 'Hit' : 'Miss'}</strong>
        </div>
      </section>

      {!compact ? (
        <section className="metric-grid">
          {metricCards.map((card) => (
            <article className="metric-card" key={card.label}>
              <card.icon size={18} aria-hidden="true" />
              <span>{card.label}</span>
              <strong>{card.value}</strong>
            </article>
          ))}
        </section>
      ) : null}

      <section className={compact ? 'trace-layout compact' : 'trace-layout'}>
        <div className="panel timeline-panel">
          <h2>Trace Timeline</h2>
          <ol className="timeline">
            {trace.events.map((event, index) => (
              <li key={`${event.stage}-${index}`}>
                <span className="timeline-dot" />
                <div>
                  <strong>{String(event.stage)}</strong>
                  <span>{String(event.status)}</span>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <div className="panel">
          <h2>Tool Calls</h2>
          {trace.tools.length === 0 ? (
            <p className="muted">No tool calls recorded for this trace.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Tool</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {trace.tools.map((tool) => (
                  <tr key={tool}>
                    <td>{tool}</td>
                    <td>recorded</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {!compact ? (
          <div className="panel sql-panel">
            <h2>Generated SQL</h2>
            <pre>{trace.generated_sql ?? 'No generated SQL for this trace.'}</pre>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function EvaluationDashboard() {
  const evalQuery = useLatestEvaluationQuery();
  const run = evalQuery.data;
  const metrics = run?.metrics ?? [];
  const chartMetrics = metrics.filter((metric) => metric.unit === 'percent');
  const pieData = Object.entries(run?.breakdown ?? {}).map(([name, value]) => ({ name, value }));

  if (evalQuery.isLoading) {
    return <div className="panel">Loading evaluation metrics...</div>;
  }

  return (
    <div className="dashboard-stack">
      <section className="trace-summary panel">
        <div>
          <span className="label">Run ID</span>
          <strong>{run?.run_id}</strong>
        </div>
        <div>
          <span className="label">Rows Evaluated</span>
          <strong>{run?.rows_evaluated}</strong>
        </div>
        <div>
          <span className="label">Datasets</span>
          <strong>{run?.datasets.length}</strong>
        </div>
      </section>

      <section className="metric-grid">
        {metrics.map((metric) => (
          <article className="metric-card" key={metric.name}>
            <ShieldCheck size={18} aria-hidden="true" />
            <span>{metric.name}</span>
            <strong>{metric.unit === 'percent' ? `${metric.value}%` : `${metric.value}s`}</strong>
          </article>
        ))}
      </section>

      <section className="eval-layout">
        <div className="panel chart-panel">
          <h2>Evaluation Metrics</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartMetrics}>
              <CartesianGrid stroke="#e7ecef" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="value" fill="#0f8b84" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel chart-panel">
          <h2>Dataset Coverage</h2>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={96}>
                {pieData.map((entry, index) => (
                  <Cell key={entry.name} fill={pieColors[index % pieColors.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
