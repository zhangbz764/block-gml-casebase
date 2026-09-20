"""Read-only PostgreSQL connection helper for Chapter 3 experiments."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = REPO_ROOT / ".env"


@contextmanager
def readonly_connection(statement_timeout_ms: int = 120_000):
    """Open a transaction that PostgreSQL itself enforces as read-only."""
    load_dotenv(ENV_PATH)
    required = ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing database environment variables: {', '.join(missing)}")

    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        application_name="ch03_phase1_readonly_audit",
    )
    try:
        conn.set_session(readonly=True, autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (statement_timeout_ms,))
        yield conn
    finally:
        conn.rollback()
        conn.close()
