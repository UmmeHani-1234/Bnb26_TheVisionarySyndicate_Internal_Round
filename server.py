"""FastAPI server application for Black Box AI Agent Observability & Debugging API."""

import logging
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import text

from api.routes import router as api_router
from storage.database import init_db, engine

logger = logging.getLogger(__name__)

# Initialize database schema on startup
init_db()

app = FastAPI(
    title="Black Box | AI Agent Debugger & Trace Storage API",
    description="Automated trace recording, retrieval, 6-signal diagnosis, checkpoint replay, and evaluation.",
    version="1.0.0",
)

# Production CORS configuration
raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints (/runs, /runs/{run_id}, /agent/run, /evaluation/..., etc.)
app.include_router(api_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
@app.get("/viewer", response_class=HTMLResponse)
def serve_viewer():
    """Serves the basic trace viewer UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h3>Black Box Trace API is active. Access dashboard on port 8501 / 8502.</h3>")


@app.get("/health")
def health_check():
    """Health check endpoint reporting service and database connection status."""
    db_connected = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_connected = True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)

    return {
        "status": "ok" if db_connected else "degraded",
        "service": "Black Box AI Debugger API",
        "database": "connected" if db_connected else "disconnected",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT") or os.getenv("PORT") or "8000")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
