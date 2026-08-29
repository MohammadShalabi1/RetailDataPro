# RetailData-Pro

RetailData-Pro is an AI Retail Intelligence Agent Platform built from scratch with a React/TypeScript frontend, a FastAPI backend, and PostgreSQL as the primary database.

This first foundation task intentionally avoids Docker, Gradio, microservices, Redis, Kafka, and AI provider setup. Later milestones will add the retail schema, analytics APIs, Gemini provider abstraction, typed tools, safe text-to-SQL, hybrid RAG, tracing, and evaluations.

## Project Structure

```text
frontend/   React, TypeScript, Vite, TanStack Query, React Router, Recharts
backend/    FastAPI, Pydantic, SQLAlchemy, Alembic
```

## Backend Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy ..\.env.example .env
uvicorn app.main:app --reload
```

The API health endpoint is available at:

```text
http://localhost:8000/api/health
```

## Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

The frontend is available at:

```text
http://localhost:5173
```

## Configuration

Configuration is environment based. Start from `.env.example` and provide local values in `.env`.

Never commit real secrets. Gemini API keys and other provider credentials must remain server-side only when later AI tasks add them.
