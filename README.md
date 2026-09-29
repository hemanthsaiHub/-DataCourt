# DataCourt: put your business decisions on trial

Most "AI analytics" tools generate a chart and a paragraph and hope you trust it. **Nobody actually does this in
the real world** — a courtroom where two AI lawyers argue opposite sides of a business decision using only your
data, a machine **checks every number they say against the data before it's allowed to count**, and only then does
an AI judge rule. That verification step, not the debate format, is the actual idea worth building: most LLM
apps let the model state facts and trust it; this one refuses to let a claim influence the outcome unless it can
be checked.

Built by Hemanth Sai Kumar Reddy, final-year B.Tech CSE project.

## How a hearing works

1. **You state a decision**, e.g. "Should we exit Furniture in the West region?", and pick a date column, a metric
   (e.g. revenue) and, optionally, which categories to compare by (region, channel, ...).
2. **The evidence engine** (`services/evidence.py`) computes everything with pandas alone — no LLM involved: the
   overall change between two time windows, which categories and category *combinations* drove it, a
   counterfactual ("without the West, the change would have been..."), concentration risk, and outlier days. Each
   fact becomes a numbered evidence item (E1, E2, ...).
3. **Two AI counsels** (`services/hearing.py`) argue for and against the decision, in an opening round and a
   rebuttal round. They may only write claims that cite an evidence id, and any number they state must match one
   in that evidence.
4. **The verifier** (`services/verifier.py`) checks every claim by re-parsing the numbers in it and comparing them
   to the cited evidence's own numbers. Claims that don't match are marked `unsupported` and struck out in the
   transcript — the judge never sees them as if they were true.
5. **An AI judge** rules using only the verified claims, and its stated confidence is capped by how much of the
   argument was actually grounded — if most claims failed verification, the ruling cannot claim to be confident,
   no matter what the model says.

## Architecture

```
Browser (public/: courtroom UI, HTML + vanilla JS)
        |  fetch /api/*
FastAPI (datacourt/main.py)
        |
services/evidence.py   -> pandas: builds the numbered evidence docket (no LLM)
services/hearing.py     -> orchestrates counsel rounds + judge (parallel per side)
services/verifier.py    -> regex-based number extraction + citation checking
services/llm.py         -> Gemini / Groq / OpenRouter client (free tiers)
        |
Vercel serverless function (api/index.py)
```

The server is stateless — the browser sends the file with every request — which is what lets it run on Vercel.

## Run on your PC

Requires Python 3.11+ and Git.

**Windows**
```bat
cd datacourt
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn datacourt.main:app --reload
```

**macOS / Linux**
```bash
cd datacourt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn datacourt.main:app --reload
```

Open http://localhost:8000, click **Use sample business data**, and try the decision
"Should we exit Furniture in the West region?" with Date = `date`, Metric = `revenue`. Without an API key, `/api/describe`
still works, but a hearing needs one (see below). API docs: http://localhost:8000/api/docs

### Tests
```bash
pip install -r requirements-full.txt
pytest -q
```
The end-to-end hearing test uses a stub model (no network needed), so CI never depends on a live API key.

## Free API keys (pick one)

| Provider | Get a key | Env variable |
|---|---|---|
| Google Gemini | https://aistudio.google.com | `GEMINI_API_KEY` |
| Groq | https://console.groq.com | `GROQ_API_KEY` |
| OpenRouter | https://openrouter.ai | `OPENROUTER_API_KEY` |

Put it in `.env` (never commit that file — it's in `.gitignore`). If a model name is retired, override it with
`GEMINI_MODEL` / `GROQ_MODEL` / `OPENROUTER_MODEL`.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: DataCourt - AI hearing engine for business decisions"
git branch -M main
git remote add origin https://github.com/hemanthsaiHub/datacourt.git
git push -u origin main
```
(Create the empty `datacourt` repo on github.com first, with no README.)

## Deploy to Vercel

**Dashboard:** vercel.com → Add New → Project → import the GitHub repo → Framework "Other" → add environment
variable `GEMINI_API_KEY` (and `LLM_PROVIDER=gemini`) → Deploy.

**CLI:**
```bash
npm i -g vercel
vercel login
vercel
vercel env add GEMINI_API_KEY
vercel --prod
```

### Limits to know
- Vercel functions here are capped at 60 seconds, and a hearing runs 5 LLM calls (2 sides × 2 rounds + judge, the
  two sides run in parallel) — comfortably inside that, but a slow provider under load could push close to it.
- Uploads are capped at ~4.5 MB on Vercel (`MAX_UPLOAD_MB` in `.env`).
- The verifier does numeric matching, not semantic fact-checking — a claim can cite a real number out of context.
  It catches hallucinated statistics, not subtler misreadings; say this plainly in a viva if asked.

## What makes this different from a normal "chat with your data" app
Most such projects let the LLM state facts about the data directly, so a hallucinated number looks exactly like a
real one. Here, **no number reaches the verdict unless a machine — not another LLM — can find it in the
evidence.** That is a real (if small-scale) example of the "verify, don't just trust" pattern used in serious
LLM-agent safety work, and it is easy to demonstrate live: try a decision, then read the transcript and point out a
struck-through `unsupported` claim to show the guardrail actually catching something.

## Ideas to extend it (good for a viva)
Add a jury mode (3-5 short LLM votes instead of one judge, to show variance); log every hearing's grounding score
over many runs to plot how reliable the AI counsels are; let the judge request more evidence instead of ruling
when grounding is low; swap the regex number-matcher for a small classifier.

## Resume bullet
Built an AI decision-hearing platform where adversarial LLM agents argue a business decision from real data,
every numeric claim is programmatically verified against the source data before use, and a judge's confidence is
capped by how well its argument was grounded — deployed serverless on Vercel with CI on GitHub Actions.
