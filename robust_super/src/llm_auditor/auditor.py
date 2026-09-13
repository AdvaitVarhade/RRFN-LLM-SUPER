import json
import re
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .prompt_builder import LLMPromptBuilder
from .cache import LLMCache
from .mock_auditor import MockLLMAuditor

class LLMAuditor:
    """
    Main LLM semantic auditor engine.
    Orchestrates pre-filtering, caching, telemetry tracking, and API / mock auditing.
    Supports both Google GenAI SDK (google.genai) and legacy google.generativeai, plus OpenAI.
    """
    def __init__(
        self,
        prompt_builder: LLMPromptBuilder,
        provider: str = "mock",
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        cache_path: str = "data/processed/llm_cache.db",
        prefilter_threshold: float = 0.60,
        fallback_score: float = 0.50
    ):
        self.prompt_builder = prompt_builder
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.prefilter_threshold = prefilter_threshold
        self.fallback_score = fallback_score
        
        self.cache = LLMCache(cache_path)
        self.mock = MockLLMAuditor()

        # Telemetry & Audit Logs for in-depth inspection
        self.telemetry: Dict[str, Any] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "est_prompt_tokens": 0,
            "errors": 0,
            "provider": provider,
            "model_name": model_name
        }
        self.audit_records: List[Dict[str, Any]] = []

    def audit_all(
        self,
        train_dict: Dict[int, List[Tuple[int, int, int]]],
        R_RRFN: Dict[Tuple[int, int], float],
        ground_truth_labels: Optional[Dict[Tuple[int, int], int]] = None,
        user_mean_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Audits all training interactions.
        Interactions with R_RRFN >= prefilter_threshold are assumed authentic (score = 0.95)
        to minimize API calls and latency. Only suspicious items are audited.
        """
        R_LLM: Dict[Tuple[int, int], float] = {}

        # 1. Identify suspicious candidate interactions
        suspicious_list: List[Tuple[int, int, int]] = []
        for u, interactions in train_dict.items():
            for item, rating, ts in interactions:
                rrfn_score = R_RRFN.get((u, item), 0.5)
                if rrfn_score < self.prefilter_threshold:
                    suspicious_list.append((u, item, rating))
                else:
                    R_LLM[(u, item)] = 0.95  # Fast-path for statistically sound interactions

        # 2. Audit suspicious interactions
        if self.provider == "mock":
            gt_labels = ground_truth_labels or {}
            means = user_mean_ratings or {}
            mock_scores = self.mock.audit_batch(suspicious_list, gt_labels, means)
            R_LLM.update(mock_scores)
            
            # Record mock sample records for dashboard walkthrough
            for u, item, rating in suspicious_list[:30]:
                profile_str = self.prompt_builder.build_user_profile_summary(train_dict.get(u, []))
                prompt = self.prompt_builder.build_audit_prompt(u, profile_str, item, rating)
                score = mock_scores.get((u, item), 0.5)
                title, genres = self.prompt_builder.item_info.get(item, (f"Movie_{item}", ["Unknown"]))
                gt = gt_labels.get((u, item), 1)
                
                self.audit_records.append({
                    "user_id": u,
                    "item_id": item,
                    "title": title,
                    "genres": genres,
                    "rating": rating,
                    "score": score,
                    "reason": f"Heuristic audit: User baseline {means.get(u, 3.0):.1f}★ vs interaction {rating}★ on {'genuine' if gt == 1 else 'injected'} item.",
                    "prompt": prompt,
                    "is_cached": False,
                    "source": "mock",
                    "ground_truth": gt
                })
        else:
            # Handle live API calls (Gemini / OpenAI) with caching
            for u, item, rating in suspicious_list:
                cached = self.cache.get(u, item, rating, self.model_name)
                profile_str = self.prompt_builder.build_user_profile_summary(train_dict.get(u, []))
                prompt = self.prompt_builder.build_audit_prompt(u, profile_str, item, rating)
                title, genres = self.prompt_builder.item_info.get(item, (f"Movie_{item}", ["Unknown"]))
                gt = (ground_truth_labels or {}).get((u, item), 1)

                if cached:
                    score, reason = cached
                    self.telemetry["cache_hits"] += 1
                    is_cached = True
                else:
                    self.telemetry["cache_misses"] += 1
                    self.telemetry["api_calls"] += 1
                    self.telemetry["est_prompt_tokens"] += len(prompt.split()) * 2
                    score, reason = self._call_live_api(u, item, rating, train_dict.get(u, []))
                    self.cache.put(u, item, rating, score, reason, self.model_name)
                    is_cached = False

                R_LLM[(u, item)] = score

                if len(self.audit_records) < 30:
                    self.audit_records.append({
                        "user_id": u,
                        "item_id": item,
                        "title": title,
                        "genres": genres,
                        "rating": rating,
                        "score": score,
                        "reason": reason,
                        "prompt": prompt,
                        "is_cached": is_cached,
                        "source": self.provider,
                        "ground_truth": gt
                    })

        return R_LLM

    def audit_single(
        self,
        user_id: int,
        item_id: int,
        rating: int,
        user_history: List[Tuple[int, int, int]],
        force_live: bool = False
    ) -> Dict[str, Any]:
        """
        Audits a single specific interaction on-demand for live UI inspection.
        """
        t0 = time.time()
        profile_str = self.prompt_builder.build_user_profile_summary(user_history)
        prompt = self.prompt_builder.build_audit_prompt(user_id, profile_str, item_id, rating)
        title, genres = self.prompt_builder.item_info.get(item_id, (f"Movie_{item_id}", ["Unknown"]))

        cached = None if force_live else self.cache.get(user_id, item_id, rating, self.model_name)
        if cached:
            score, reason = cached
            is_cached = True
        else:
            if self.provider == "mock":
                u_mean = float(np.mean([r for _, r, _ in user_history])) if user_history else 3.5
                score, reason = self.mock.audit_interaction(user_id, item_id, rating, ground_truth_label=1, user_mean_rating=u_mean)
            else:
                score, reason = self._call_live_api(user_id, item_id, rating, user_history)
                self.cache.put(user_id, item_id, rating, score, reason, self.model_name)
            is_cached = False

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "user_id": user_id,
            "item_id": item_id,
            "title": title,
            "genres": genres,
            "rating": rating,
            "score": score,
            "reason": reason,
            "prompt": prompt,
            "is_cached": is_cached,
            "latency_ms": latency_ms,
            "provider": self.provider,
            "model_name": self.model_name
        }

    def _call_live_api(self, user_id: int, item_id: int, rating: int, user_history: List[Tuple[int, int, int]]) -> Tuple[float, str]:
        """Calls live LLM provider API with fallback to heuristic scoring."""
        profile_str = self.prompt_builder.build_user_profile_summary(user_history)
        prompt = self.prompt_builder.build_audit_prompt(user_id, profile_str, item_id, rating)

        try:
            if self.provider == "gemini":
                # 1. Try modern google.genai SDK
                try:
                    from google import genai
                    client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                    return self._parse_json_response(response.text)
                except Exception as e_genai:
                    # 2. Fallback to google.generativeai legacy
                    import google.generativeai as legacy_genai
                    if self.api_key:
                        legacy_genai.configure(api_key=self.api_key)
                    model = legacy_genai.GenerativeModel(self.model_name)
                    response = model.generate_content(prompt)
                    return self._parse_json_response(response.text)

            elif self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return self._parse_json_response(response.choices[0].message.content)
            else:
                return self.fallback_score, f"Unsupported provider: {self.provider}"
        except Exception as e:
            self.telemetry["errors"] += 1
            return self.fallback_score, f"API error: {str(e)}"

    def _parse_json_response(self, text: str) -> Tuple[float, str]:
        """Robust JSON extraction from LLM response text."""
        try:
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("semantic_reliability", self.fallback_score))
                reason = str(data.get("reason", ""))
                return max(0.0, min(1.0, score)), reason
        except Exception:
            pass
        return self.fallback_score, "Fallback: could not parse JSON response"
