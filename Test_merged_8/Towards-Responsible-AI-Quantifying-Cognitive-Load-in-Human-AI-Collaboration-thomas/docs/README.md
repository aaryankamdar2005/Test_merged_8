# CogniTrack AI

CogniTrack is organized as a FastAPI backend with a browser frontend.

## Project structure

- `frontend/` — HTML, CSS, browser JavaScript, and tracking modules.
- `backend/` — FastAPI application, PostgreSQL storage adapter, eye-tracking modules, the separate facial-expression ZIP module, and its ONNX model.
- `backend/models/` — the facial-expression ONNX model required by the separate facial endpoint.
- `tests/` — API and analysis tests.
- `docs/` — project and testing documentation.
- `scripts/` — Windows setup and backend launch scripts.
- `data/` — local runtime data and migration backups; do not share this folder.
- `.env` — private local secrets; do not share this file.

## Storage

For local development, the checked-in configuration uses a SQLite database in
`data/cognitrack.db`; it is created automatically on first startup:

```env
DATABASE_URL=sqlite:///data/cognitrack.db
```

To store assessment and camera-analysis data in PostgreSQL, replace it with a
working database URL:

```env
DATABASE_URL=postgresql://cognitrack:your-password@127.0.0.1:5432/AtharvaDB
```

Tables are created automatically on first use with either storage backend.

### RAG vector search

The RAG pipeline uses PostgreSQL-native vector search through the `pgvector`
extension. The application adds an `embedding_vector vector(384)` column to
`rag_chunks`, migrates existing JSON embeddings into it, removes the old JSON
column, creates a cosine HNSW index, and uses pgvector for semantic retrieval.
The application keeps the existing 45% BM25 / 55% semantic hybrid reranking.

RAG observability is enabled with `RAG_EVALUATOR=native`. Each request follows
`pgvector -> BM25 hybrid ranking -> feature reranker -> LLM`, and persists a
`trace_id` with retrieval, reranking, generation, and evaluation timings in
`rag_pipeline` and `rag_prompt_evaluations`. Evaluation records context
precision, context recall (when a reference answer is available), faithfulness,
answer relevancy, and mean retrieval score. The default evaluator is local and
deterministic; it is intentionally labelled `native`, not RAGAS or DeepEval.
This avoids claiming third-party evaluation when those packages and a reviewed
evaluation dataset are not installed.

`RAG_VECTOR_BACKEND=pgvector` is required. The previous Python-only semantic
retrieval fallback has been removed, so startup fails clearly if pgvector is
not installed or PostgreSQL is unreachable.

The project does not use ChromaDB, FAISS, Pinecone, or Qdrant. PostgreSQL
stores the document/chunk metadata and RAG pipeline records, while the
`pgvector` extension is the indexed retrieval layer inside that same database.

### OCR for scanned PDFs

PDF ingestion first uses `pypdf` for normal text-based pages. If a page has no
extractable text, the backend renders that page with PyMuPDF and sends the
image to local Tesseract through `pytesseract`. OCR text then follows the same
chunking, embedding, pgvector retrieval, hybrid ranking, reranking, and
evaluation path as normal PDF text. The `rag_documents` record stores
`ocr_page_count` and `ocr_pages` so the source of extracted text is auditable.
Install the Python packages from `requirements.txt` and install the Tesseract
OCR application on Windows. Set `TESSERACT_CMD` if it is not on PATH. Set
`OCR_ENABLED=false` only when scanned PDFs are not required.
The same status endpoint reports `ocr.ready`; it must be `true` before uploading
scanned PDFs. Text-based PDFs continue to work when OCR is unavailable.

On Windows, install the pgvector extension for the exact PostgreSQL major
version used by the project, then enable it in the application database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Restart the backend and check `/api/rag/status`; the response should report
`"vector_backend": "pgvector"`.

On the included Windows/PostgreSQL 18 setup, run `./scripts/setup_postgres.ps1` from the
project folder. It securely prompts for the local `postgres` administrator password,
creates the application database and role with a generated password, and writes only
the application connection URL to the ignored `.env` file.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Also replace both example role passwords. The backend refuses to start when any
admin or host credential is missing. Role access tokens expire after
`TOKEN_TTL_MINUTES` (60 by default).

## Run backend

```powershell
uvicorn main:app --app-dir backend --reload --host 127.0.0.1 --port 8002
```

The backend is a Python/FastAPI application, so do not run `npm run dev` from
`backend/`. From the project folder you can instead use:

```powershell
npm run dev:backend
```

This command uses `venv\\Scripts\\python.exe`, created by the setup step. If
you copied the project from another computer, recreate the virtual environment
instead of reusing its copied `venv` or `.venv` folder:

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

API health check:

```text
http://127.0.0.1:8002/api/health
```

Swagger documentation:

```text
http://127.0.0.1:8002/docs
```

## Run the combined app

From the project folder, run `start.bat`. FastAPI serves both the frontend and
the API at `http://127.0.0.1:8002/`.

## Run the frontend development server

The Vite development server is optional: FastAPI already serves the frontend
when the backend is running. To use Vite with live frontend reloads, open a
second terminal in the project folder and run:

```powershell
cd frontend
npm install
npm run dev
```

It starts at `http://127.0.0.1:5173` and proxies `/api` requests to the backend
at port 8002. To launch both development servers from the project folder, run
`npm install` once there and then `npm run dev`.

## Notes

- Keep `.env` private and never upload it to GitHub.
- Admin and Host credentials stay in `.env`, access tokens stay only in server memory,
  and their login, logout, dashboard views, and page activity are never persisted to
  PostgreSQL. Their dashboards are read-only views of participant assessment data.
- The Host page is a live operational view of participant progress, activity, alerts,
  prompt-writing status/length/AI-request count, and specialist-agent reports. The Admin page is a stored-results view with participant
  comparisons, tracking-coverage and LLM-usage graphs, and protected CSV downloads
  for participant, answer, eye, and keyboard data.
- ChatGPT and Groq are supported by `/api/llm/chat/stream`. Both require their
  corresponding API keys in `.env`; Groq is selected automatically for assessment questions.
- Coding & Programming tasks include Java and Python editors with a Run Code button. Code is sent
  to the external Paiza.IO sandbox through `/api/code/run`; it is never executed on
  the CogniTrack server. Set `PAIZA_API_BASE_URL` in `.env` only if using a compatible
  self-hosted or alternative Paiza-compatible service.
- Browser camera frames are sampled continuously. MediaPipe eye, gaze, blink,
  fatigue, and head-pose measurements are saved in `eye_tracking`.
- The separate facial-expression endpoint uses the supplied ONNX model and saves
  one-second facial-expression samples in the `facial_expression` component.
  Its emotion scores are shown only in the Admin Facial Expression table and CSV.
- Eye tracking starts automatically after the participant clicks Start Assessment and
  grants camera permission. There is no eye-calibration screen or calibration step;
  pupil coordinates and continuing eye measurements are saved in `eye_tracking`.
  Camera permission is mandatory; non-local devices require HTTPS.
- Keyboard behavior is measured without recording actual key values or answer text.
  Idle autosaves and final submission upsert one `keyboard_tracking` row per
  participant and question, including speed, pauses, corrections, paste count,
  response latency, and answer length metrics.
- Active participants send a progress heartbeat every ten seconds. The authenticated
  Admin dashboard provides platform-wide participant, session, answer, chat, keyboard,
  vision and monitoring information.
- Host is an orchestrator agent, not a second administrator. It dispatches each
  heartbeat to Progress, Activity, Technical, Wellbeing, and Summary specialist agents,
  then consolidates their explainable findings. Agents may flag inactivity, repeated
  tab switching, missing vision telemetry, and high fatigue, but cannot score, accuse,
  reject, terminate, or penalize participants.
- Camera access from another device requires HTTPS; browsers allow HTTP camera access
  only on localhost. Assessment entry still works when camera permission is unavailable.

### SQL Expert Agent

Authenticated Admin, Host, and Dashboard users can ask natural-language analytics
questions through `POST /api/dashboard/sql-agent`:

```json
{"question":"Which participants completed the assessment?","provider":"Groq"}
```

The agent receives the live approved application schema, asks the configured ChatGPT
or Groq model to propose one query, validates that query as a single read-only
`SELECT`/`WITH` statement, caps results at 200 rows, and executes it through the
application storage adapter. Destructive statements, comments, multiple statements,
unknown tables, unknown columns, wildcard projections, sensitive columns, blocked
database functions, and non-integer or oversized limits are rejected. PostgreSQL
execution uses a read-only transaction and a five-second statement timeout. The
response includes the generated SQL, a short explanation, model metadata, and
returned rows. Participant sessions cannot access this endpoint.

## Tests

From the project folder:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```
