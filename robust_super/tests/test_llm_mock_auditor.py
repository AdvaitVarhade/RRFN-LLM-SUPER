import pytest
import pandas as pd
from src.llm_auditor.mock_auditor import MockLLMAuditor
from src.llm_auditor.prompt_builder import LLMPromptBuilder
from src.llm_auditor.auditor import LLMAuditor

def test_mock_llm_auditor():
    movies_df = pd.DataFrame({
        "item_id": [0, 1, 2],
        "title": ["Matrix", "Toy Story", "Inception"],
        "genres": [["Action", "Sci-Fi"], ["Animation", "Children"], ["Action", "Sci-Fi"]]
    })
    prompt_builder = LLMPromptBuilder(movies_df)

    user_history = [(0, 5, 1000), (2, 5, 1010)]
    profile_summary = prompt_builder.build_user_profile_summary(user_history)
    assert "Preferred Genres" in profile_summary

    prompt = prompt_builder.build_audit_prompt(user_id=1, user_profile_summary=profile_summary, item_id=1, observed_rating=1)
    assert "semantic_reliability" in prompt

    mock = MockLLMAuditor(seed=42)
    score_gen, reason_gen = mock.audit_interaction(user_id=1, item_id=0, rating=5, ground_truth_label=1)
    score_inj, reason_inj = mock.audit_interaction(user_id=1, item_id=1, rating=1, ground_truth_label=0)

    assert score_gen > score_inj
    assert 0.0 <= score_gen <= 1.0
    assert 0.0 <= score_inj <= 1.0

    # Test LLMAuditor integration & audit_single
    auditor = LLMAuditor(prompt_builder=prompt_builder, provider="mock")
    train_dict = {1: user_history}
    single_res = auditor.audit_single(user_id=1, item_id=0, rating=5, user_history=user_history)
    assert "score" in single_res
    assert "prompt" in single_res
    assert "latency_ms" in single_res
    assert single_res["score"] >= 0.5

