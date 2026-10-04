import os
import math
import pickle
import pytest
import concurrent.futures
import pandas as pd
from src.llm_auditor.cache import LLMCache
from src.llm_auditor.auditor import LLMAuditor, redact_secrets
from src.llm_auditor.prompt_builder import LLMPromptBuilder

def test_cache_memory_database_persistence():
    """Verify that in-memory SQLite database persists across operations and handles queries."""
    cache = LLMCache(":memory:")
    for i in range(10):
        cache.put(i, i * 2, 5, 0.85, f"reason_{i}", "gemini-3.6-flash")

    for i in range(10):
        res = cache.get(i, i * 2, 5, "gemini-3.6-flash")
        assert res is not None
        assert res[0] == 0.85
        assert res[1] == f"reason_{i}"

    assert cache.get(999, 999, 5, "gemini-3.6-flash") is None
    cache.close()

def test_cache_connection_closure_and_file_release(tmp_path):
    """Verify that disk connections are closed cleanly in finally block, avoiding Windows WinError 32."""
    db_file = str(tmp_path / "test_conn_leak.db")
    cache = LLMCache(db_file)
    cache.put(1, 10, 5, 0.9, "Great movie", "model_v1")
    hit = cache.get(1, 10, 5, "model_v1")
    assert hit is not None
    assert hit[0] == 0.9

    # Because conn.close() was executed in finally, we can safely open/inspect/remove the DB file
    cache.close()
    if os.path.exists(db_file):
        os.remove(db_file)
    assert not os.path.exists(db_file)

def test_cache_wal_mode_pragmas(tmp_path):
    """Verify PRAGMA journal_mode=WAL, synchronous=NORMAL, and busy_timeout=30000 on disk DB."""
    db_file = str(tmp_path / "test_pragmas.db")
    cache = LLMCache(db_file)
    with cache._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0]
        cursor.execute("PRAGMA synchronous;")
        synchronous = cursor.fetchone()[0]
        cursor.execute("PRAGMA busy_timeout;")
        busy_timeout = cursor.fetchone()[0]

    assert str(journal_mode).upper() == "WAL"
    # synchronous: 1 corresponds to NORMAL
    assert synchronous in (1, "1", "NORMAL")
    assert busy_timeout >= 30000
    cache.close()

def test_cache_score_bounds_validation():
    """Verify semantic reliability score is strictly bounded in [0.0, 1.0] and rejects NaN/inf."""
    cache = LLMCache(":memory:")
    # Valid boundaries
    cache.put(1, 1, 1, 0.0, "min bound", "m")
    cache.put(1, 2, 1, 1.0, "max bound", "m")
    cache.put(1, 3, 1, 0.5, "mid bound", "m")
    assert cache.get(1, 1, 1, "m")[0] == 0.0
    assert cache.get(1, 2, 1, "m")[0] == 1.0
    assert cache.get(1, 3, 1, "m")[0] == 0.5

    # Invalid values
    for bad in [-0.001, -10.0, 1.001, 5.0, float("nan"), float("inf"), float("-inf"), "invalid"]:
        with pytest.raises(ValueError):
            cache.put(2, 2, 2, bad, "invalid", "m")
    cache.close()

def test_cache_multithreaded_concurrency(tmp_path):
    """Verify SQLite cache thread safety under concurrent writes and reads with 10 threads."""
    db_file = str(tmp_path / "test_threads.db")
    cache = LLMCache(db_file)

    def worker_action(idx: int):
        score = 0.5 + (idx % 50) / 100.0
        cache.put(idx, idx * 3, 4, score, f"thread_reason_{idx}", "thread_model")
        read_val = cache.get(idx, idx * 3, 4, "thread_model")
        return read_val

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker_action, range(60)))

    assert len(results) == 60
    for idx, res in enumerate(results):
        assert res is not None
        expected_score = 0.5 + (idx % 50) / 100.0
        assert math.isclose(res[0], expected_score, abs_tol=1e-5)
        assert res[1] == f"thread_reason_{idx}"
    cache.close()

def test_secret_redactor():
    """Verify redact_secrets strips Gemini keys, OpenAI keys, URL parameters, Bearer tokens, and secrets."""
    # Gemini API Key
    gemini_msg = "Error 403 on request to endpoint?key=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q and header"
    cleaned = redact_secrets(gemini_msg)
    assert "AIzaSy" not in cleaned
    assert "[REDACTED_GEMINI_KEY]" in cleaned or "[REDACTED_KEY]" in cleaned

    # OpenAI API Key
    openai_msg = "Failed auth: sk-proj-12345678901234567890123456789012345"
    cleaned_openai = redact_secrets(openai_msg)
    assert "sk-proj-" not in cleaned_openai
    assert "[REDACTED_OPENAI_KEY]" in cleaned_openai

    # URL query parameter
    url_msg = "https://api.example.com/query?access_token=secret_token_12345&other=1"
    cleaned_url = redact_secrets(url_msg)
    assert "secret_token_12345" not in cleaned_url
    assert "[REDACTED_KEY]" in cleaned_url

    # Bearer token
    bearer_msg = "Authorization: Bearer my_secret_bearer_token_12345678"
    cleaned_bearer = redact_secrets(bearer_msg)
    assert "my_secret_bearer" not in cleaned_bearer
    assert "[REDACTED_TOKEN]" in cleaned_bearer

    # Custom explicit secret
    custom_secret = "my_custom_production_password_999"
    msg = f"Connecting with {custom_secret} failed!"
    cleaned_custom = redact_secrets(msg, secret=custom_secret)
    assert custom_secret not in cleaned_custom
    assert "[REDACTED_API_KEY]" in cleaned_custom

def test_auditor_pickling_and_secret_scrubbing():
    """Verify LLMAuditor scrubs api_key during pickling and via scrub_secrets method."""
    movies_df = pd.DataFrame({"item_id": [0], "title": ["Test Movie"], "genres": [["Action"]]})
    pb = LLMPromptBuilder(movies_df)
    auditor = LLMAuditor(pb, provider="mock", api_key="AIzaSySecretTokenShouldNeverBePickled")

    # Pickling protection
    pickled = pickle.dumps(auditor)
    assert b"AIzaSySecretToken" not in pickled, "Plaintext API key leaked into pickled stream!"
    restored = pickle.loads(pickled)
    assert restored.api_key is None, "Restored auditor retained API key!"

    # In-place scrub method
    auditor2 = LLMAuditor(pb, provider="mock", api_key="sk-live-secret-test-token-12345")
    assert auditor2.api_key is not None
    auditor2.scrub_secrets()
    assert auditor2.api_key is None
