"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    SQLite is the supported durable backend for the lab. The connection is
    intentionally kept on the saver object for the lifetime of the graph.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite persistence requires the optional dependency: "
                "pip install 'langgraph-checkpoint-sqlite>=2.0'"
            ) from exc

        raw_url = database_url or "outputs/langgraph_checkpoints.sqlite"
        if raw_url.startswith("sqlite:///"):
            raw_url = raw_url[10:]
        if raw_url != ":memory:":
            db_path = Path(raw_url)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            raw_url = str(db_path)
        connection = sqlite3.connect(raw_url, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.commit()
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise RuntimeError(
            "Postgres persistence is not enabled in this lab build; use kind='sqlite' "
            "or install and configure a project-specific Postgres adapter."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
