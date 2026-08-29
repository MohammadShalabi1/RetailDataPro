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

## Database Setup

Create a PostgreSQL database locally or in a managed PostgreSQL service, then set `DATABASE_URL` in `backend/.env`.

Run migrations:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
alembic upgrade head
```

Seed realistic development data:

```powershell
python -m app.seed.run
```

The seed command creates customers, categories, products, suppliers, orders, order items, inventory events, sources, source chunks, conversations, messages, and AI trace metadata. It resets existing seed tables in development mode, so review `APP_ENV` before running it against any shared database.

## Analytics API

After running migrations and seed data, deterministic analytics endpoints are available under:

```text
GET /api/analytics/revenue
GET /api/analytics/sales-trends?interval=month
GET /api/analytics/top-products?limit=10
GET /api/analytics/top-customers?limit=10
GET /api/analytics/category-performance?limit=10
GET /api/analytics/inventory?limit=10
GET /api/analytics/supplier-performance?limit=10
```

Most analytics endpoints accept optional `start_date` and `end_date` ISO date query parameters. Ranked endpoints accept `limit` from `1` to `100`.

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

## AI Provider Boundary

The backend owns AI provider access. Configure Gemini only in `backend/.env`:

```text
GEMINI_API_KEY=your-local-key
GEMINI_TEXT_MODEL=gemini-3.5-flash-lite
GEMINI_STRUCTURED_MODEL=gemini-3.6-flash
AI_PROVIDER_TIMEOUT_SECONDS=30
AI_PROVIDER_MAX_RETRIES=2
GEMINI_35_FLASH_LITE_INPUT_COST_PER_1M=0.30
GEMINI_35_FLASH_LITE_OUTPUT_COST_PER_1M=2.50
GEMINI_36_FLASH_INPUT_COST_PER_1M=1.50
GEMINI_36_FLASH_OUTPUT_COST_PER_1M=7.50
```

`gemini-3.5-flash-lite` is the default for simple text generation because it is fast and cost-efficient. `gemini-3.6-flash` is the default for structured generation because it is stronger for agentic and code-adjacent tasks. Frontend code must never receive provider keys. AI routes and orchestration are intentionally not added until later tasks.

## Typed Routing

The backend includes a structured router for classifying questions before orchestration is added. Route categories are:

```text
conversation
retail_analytics
document_search
website_search
multi_source
```

The router uses the configured structured model through the AI provider abstraction and falls back to deterministic keyword routing if the model response cannot be validated after one repair attempt.

## Model Selection

The backend includes a lightweight model router for choosing configured Gemini models by task:

```text
simple_classification  -> GEMINI_TEXT_MODEL
analytics_answer       -> GEMINI_TEXT_MODEL
structured_generation  -> GEMINI_STRUCTURED_MODEL
multi_source_synthesis -> GEMINI_STRUCTURED_MODEL
```

Cost estimates use model-specific Decimal pricing settings. If a configured model has no pricing entry, the estimate is marked unavailable instead of pretending the request is free.

## Agent Orchestration Boundary

The backend includes an internal `run_turn(...)` orchestration skeleton. Its stage order is:

```text
load_context
apply_input_policy
select_route
plan_execution
select_model
authorize_tools
execute_tools
build_context
generate_answer
validate_answer
finalize_trace
```

The input policy runs before model calls. Tool execution is intentionally skipped until the typed tool gateway is implemented.

## Typed Tool Gateway

All future model-requested tool calls must pass through the backend gateway:

```python
authorize_and_execute_tool(...)
```

Registered tools are:

```text
analytics_summary
retail_sql
document_search
website_search
```

Only `analytics_summary` is executable today. SQL, document, and website tools are registered but return safe unavailable results until their dedicated phases are implemented.
