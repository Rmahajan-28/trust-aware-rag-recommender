from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .recommender import Recommender

app = FastAPI(title="Trust-Aware Educational Resource Recommender")
rec = Recommender()
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/api/health")
def health():
    return {"status": "ok", "resources": len(rec.resources)}


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), k: int = 5, kind: str | None = None):
    return {"query": q, "results": rec.search(q, k=k, kind=kind)}


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
