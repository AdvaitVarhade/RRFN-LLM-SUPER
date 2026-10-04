"""
robust_super/conftest.py
========================
Pytest configuration and test infrastructure harness for RRFN-LLM-SUPER.

Provides:
  1. Namespace collision resolution: prepends robust_super to sys.path[0]
     so that 'from src...' resolves to 'robust_super/src' rather than root 'Project1/src'.
  2. Backward compatibility: appends PROJECT_ROOT so 'from robust_super.src...' resolves.
  3. Module cache isolation: evicts any stale 'src' package entries from sys.modules.
  4. Global fast synthetic mode for tests via ROBUST_SUPER_SYNTHETIC=1.
  5. Reusable test fixtures for CPU execution and synthetic datasets.
"""
import os
import sys
import pytest

# ── 1. Path Resolution & Namespace Isolation ─────────────────────────────────
ROBUST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROBUST_DIR)

# Prepend robust_super to sys.path so 'import src...' finds robust_super/src first
if ROBUST_DIR in sys.path:
    sys.path.remove(ROBUST_DIR)
sys.path.insert(0, ROBUST_DIR)

# Append PROJECT_ROOT so 'import robust_super.src...' continues to resolve
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Evict any incorrectly cached 'src' module that does not reside in robust_super
if "src" in sys.modules:
    cached_src = sys.modules["src"]
    src_file = getattr(cached_src, "__file__", None)
    if src_file and not os.path.abspath(src_file).startswith(ROBUST_DIR):
        del sys.modules["src"]

# Set default fast synthetic mode for test runs
os.environ.setdefault("ROBUST_SUPER_SYNTHETIC", "1")


# ── 2. Shared Pytest Fixtures ────────────────────────────────────────────────
@pytest.fixture(scope="session")
def cpu_device() -> str:
    """Provides a consistent CPU device string for fast, deterministic unit tests."""
    return "cpu"


@pytest.fixture
def synthetic_movielens_data():
    """
    Returns synthetic micro-data (ratings_df, movies_df, users_df)
    generated in memory in <5ms without touching disk.
    """
    from src.data.loader import MovieLensLoader
    loader = MovieLensLoader(data_dir="", use_synthetic=True)
    return loader.load_data()


@pytest.fixture
def synthetic_clean_split(synthetic_movielens_data):
    """
    Returns a preprocessed DataSplit object based on synthetic micro-data.
    """
    from src.data.preprocessor import preprocess_dataset
    ratings_df, movies_df, users_df = synthetic_movielens_data
    return preprocess_dataset(ratings_df, movies_df, users_df)
