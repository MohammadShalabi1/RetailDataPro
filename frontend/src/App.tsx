import { Activity, Database, LineChart, ShieldCheck } from 'lucide-react';
import {
  CartesianGrid,
  Line,
  LineChart as RevenueLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { useHealthQuery } from './api/health';

const previewData = [
  { month: 'Apr', revenue: 182 },
  { month: 'May', revenue: 214 },
  { month: 'Jun', revenue: 208 },
  { month: 'Jul', revenue: 241 },
  { month: 'Aug', revenue: 268 },
];

export function App() {
  const healthQuery = useHealthQuery();
  const health = healthQuery.data;

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <nav className="topbar" aria-label="Primary navigation">
          <div className="brand">
            <Database size={24} aria-hidden="true" />
            <span>RetailData-Pro</span>
          </div>
          <div className="status-pill" data-state={health?.status ?? 'loading'}>
            <Activity size={16} aria-hidden="true" />
            <span>{healthQuery.isLoading ? 'Checking API' : health?.status ?? 'Unavailable'}</span>
          </div>
        </nav>

        <div className="hero-content">
          <div className="hero-copy">
            <h1>AI retail intelligence workspace</h1>
            <p>
              A foundation for safe text-to-SQL, hybrid retrieval, typed tools, and grounded retail
              analytics over PostgreSQL.
            </p>
          </div>

          <div className="summary-panel" aria-label="System foundation status">
            <div className="summary-row">
              <ShieldCheck size={20} aria-hidden="true" />
              <div>
                <span className="label">Backend service</span>
                <strong>{health?.service ?? 'retaildata-pro-api'}</strong>
              </div>
            </div>
            <div className="summary-row">
              <Database size={20} aria-hidden="true" />
              <div>
                <span className="label">PostgreSQL config</span>
                <strong>{health?.database_configured ? 'Configured' : 'Pending'}</strong>
              </div>
            </div>
            <div className="summary-row">
              <LineChart size={20} aria-hidden="true" />
              <div>
                <span className="label">Frontend stack</span>
                <strong>React, Router, Query, Recharts</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace-preview" aria-label="Retail analytics preview">
        <div>
          <h2>Revenue trend preview</h2>
          <p>Static sample data for the shell only; real analytics arrive in a later task.</p>
        </div>
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={260}>
            <RevenueLineChart data={previewData} margin={{ top: 20, right: 24, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="#dfe7ef" strokeDasharray="4 4" />
              <XAxis dataKey="month" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} width={42} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="revenue"
                stroke="#1d7f74"
                strokeWidth={3}
                dot={{ fill: '#1d7f74', r: 4 }}
              />
            </RevenueLineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </main>
  );
}
