"""FastAPI application. Runs locally with uvicorn and on Vercel as a serverless function (api/index.py)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import MAX_UPLOAD_MB
from .services import llm
from .services.data_loader import load_dataframe
from .services.evidence import describe
from .services.hearing import run_hearing

app = FastAPI(title="DataCourt", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
api = APIRouter(prefix="/api")


@app.exception_handler(ValueError)
async def _bad_request(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(llm.LLMUnavailable)
async def _no_llm(_: Request, exc: llm.LLMUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(llm.LLMError)
async def _llm_failed(_: Request, exc: llm.LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


async def _df(file: UploadFile):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB:g} MB. Upload a smaller sample.")
    return load_dataframe(raw, file.filename or "")


@api.get("/health")
def health():
    return {"status": "ok", "version": __version__, "llm_provider": llm.active_provider(), "max_upload_mb": MAX_UPLOAD_MB}


@api.post("/describe")
async def describe_route(file: UploadFile = File(...)):
    return describe(await _df(file))


@api.post("/hearing")
async def hearing_route(file: UploadFile = File(...), decision: str = Form(...), date_col: str = Form(...),
                        metric_col: str = Form(...), dims: str = Form(""), period_days: int = Form(0)):
    dim_list = [d.strip() for d in dims.split(",") if d.strip()] or None
    return run_hearing(await _df(file), decision, date_col, metric_col, dim_list, period_days or None)


app.include_router(api)

# Local development serves the UI from ./public. On Vercel the CDN serves /public directly.
_public = Path(__file__).resolve().parent.parent / "public"
if _public.exists():
    app.mount("/", StaticFiles(directory=_public, html=True), name="ui")
