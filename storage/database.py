"""Database connection and session factory for trace storage."""

import logging
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

Base = declarative_base()


def get_database_url() -> str:
    """Returns database connection URL, prioritizing DATABASE_URL environment variable."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return "sqlite:///traces.db"
    # Ensure psycopg 3 driver for PostgreSQL if plain postgresql:// is provided
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def create_db_engine():
    """Initializes SQLAlchemy engine with PostgreSQL or SQLite fallback."""
    url = get_database_url()
    try:
        if "sqlite" in url:
            engine = create_engine(url, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(url, pool_pre_ping=True)
        # Test connection
        with engine.connect():
            pass
        return engine
    except Exception as exc:
        logger.warning(
            "Could not connect to configured DATABASE_URL (%s). Falling back to SQLite 'traces.db'. Error: %s",
            url,
            exc,
        )
        return create_engine("sqlite:///traces.db", connect_args={"check_same_thread": False})


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency helper for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
