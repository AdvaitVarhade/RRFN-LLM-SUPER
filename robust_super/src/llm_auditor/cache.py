import sqlite3
import hashlib
import time
import os
from typing import Optional, Tuple

class LLMCache:
    """
    Persistent SQLite cache for LLM audit responses.
    Prevents repeated API calls across experiment runs.
    """
    def __init__(self, db_path: str = "data/processed/llm_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
        key = self._generate_key(user_id, item_id, rating, model_name)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_cache
                (cache_key, user_id, item_id, rating, semantic_reliability, reason, model_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key, user_id, item_id, rating, score, reason, model_name, time.time())
            )
            conn.commit()
