import sqlite3
import hashlib
import time
import os
import math
import contextlib
import threading
from typing import Optional, Tuple, Generator, Dict, Any

class LLMCache:
    """
    Persistent SQLite cache for LLM audit responses.
    Prevents repeated API calls across experiment runs.
    Hardened with connection lifecycle safety, WAL mode, thread-safe mutex,
    score bounds validation, serialization support, and persistent in-memory database support.
    """
    def __init__(self, db_path: str = "data/processed/llm_cache.db"):
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:")
        self._lock = threading.Lock()
        self._mem_conn: Optional[sqlite3.Connection] = None

        if self._is_memory:
            # Persistent connection for in-memory database across queries
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            self._mem_conn.execute("PRAGMA busy_timeout=30000;")
        else:
            abs_path = os.path.abspath(db_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        self._init_db()

    def __getstate__(self) -> Dict[str, Any]:
        """Excludes unpicklable thread locks and connection handles upon serialization."""
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_mem_conn"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Re-initializes thread locks and in-memory connection handles upon deserialization."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        if self._is_memory:
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            self._mem_conn.execute("PRAGMA busy_timeout=30000;")
            self._init_db()

    @contextlib.contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Safe connection context manager guaranteeing conn.close() in a finally block
        for filesystem databases, while keeping the persistent connection alive for :memory:.
        Configures PRAGMAs on every new file connection.
        """
        if self._is_memory:
            if self._mem_conn is None:
                raise RuntimeError("In-memory SQLite connection is closed.")
            yield self._mem_conn
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                yield conn
            finally:
                conn.close()

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_cache (
                        cache_key TEXT PRIMARY KEY,
                        user_id INTEGER,
                        item_id INTEGER,
                        rating INTEGER,
                        semantic_reliability REAL,
                        reason TEXT,
                        model_name TEXT,
                        created_at REAL
                    )
                """)
                conn.commit()

    def _generate_key(self, user_id: int, item_id: int, rating: int, model_name: str) -> str:
        raw_str = f"{user_id}|{item_id}|{rating}|{model_name}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(self, user_id: int, item_id: int, rating: int, model_name: str) -> Optional[Tuple[float, str]]:
        key = self._generate_key(user_id, item_id, rating, model_name)
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT semantic_reliability, reason FROM audit_cache WHERE cache_key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    return float(row[0]), str(row[1])
        return None

    def put(self, user_id: int, item_id: int, rating: int, score: float, reason: str, model_name: str):
        # Enforce score bounds validation: 0.0 <= score <= 1.0
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid score value: {score}. Score must be a float between 0.0 and 1.0.")

        if math.isnan(score_val) or math.isinf(score_val) or not (0.0 <= score_val <= 1.0):
            raise ValueError(f"Score must be between 0.0 and 1.0 inclusive, got {score}")

        key = self._generate_key(user_id, item_id, rating, model_name)
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audit_cache
                    (cache_key, user_id, item_id, rating, semantic_reliability, reason, model_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key, int(user_id), int(item_id), int(rating), score_val, str(reason), str(model_name), time.time())
                )
                conn.commit()

    def clear(self):
        """Clear all entries in the audit cache."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM audit_cache")
                conn.commit()

    def close(self):
        """Close persistent resources (e.g. in-memory connection)."""
        with self._lock:
            if self._mem_conn is not None:
                try:
                    self._mem_conn.close()
                except Exception:
                    pass
                self._mem_conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
