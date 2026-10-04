"""FastAPI server application for Black Box Phase 3."""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from storage.database import init_db

# Initialize database schema on startup
init_db()

app = FastAPI(
    title="Black Box | Agent Trace Storage & Retrieval API",
    description="Phase 3 MVP: Automated trace recording, retrieval, and observable step inspection.",
    version="0.3.0",
)

# Configure allowed CORS origins from environment variable or allow all
cors_env = os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGINS")
allowed_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()] if cors_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints (/runs, /runs/{run_id}, /runs/{run_id}/events, /agent/run)
app.include_router(api_router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
@app.get("/viewer", response_class=HTMLResponse)
def serve_viewer():
    """Serves the basic trace viewer UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h3>Trace viewer template not found.</h3>")


@app.get("/health")
def health_check():
    """Health check endpoint verifying DB connection and API status."""
    from storage.database import engine
    from sqlalchemy import text

    db_status = "healthy"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "phase": 3,
        "service": "Black Box Trace API",
        "database": db_status,
        "environment": os.getenv("ENV", "production"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
