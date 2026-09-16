import json
import re
import os
import time
import concurrent.futures
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .prompt_builder import LLMPromptBuilder
from .cache import LLMCache
from .mock_auditor import MockLLMAuditor

class LLMAuditor:
    """
    Main LLM semantic auditor engine.
    Orchestrates pre-filtering, caching, parallel concurrency, and API / mock auditing.
    Supports both modern Google GenAI SDK (google.genai) with gemini-3.6-flash, legacy google.generativeai, and OpenAI.
    """
    def __init__(
        self,
        prompt_builder: LLMPromptBuilder,
        provider: str = "mock",
        model_name: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
        cache_path: str = "data/processed/llm_cache.db",
        prefilter_threshold: float = 0.60,
        fallback_score: float = 0.50,
        max_live_calls: int = 20
    ):
        self.prompt_builder = prompt_builder
        self.provider = provider.lower().strip()
        self.model_name = self._normalize_model_name(self.provider, model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.prefilter_threshold = prefilter_threshold
        self.fallback_score = fallback_score
        self.max_live_calls = max_live_calls
        
        self.cache = LLMCache(cache_path)
        self.mock = MockLLMAuditor()

        # Telemetry & Audit Logs for in-depth inspection
        self.telemetry: Dict[str, Any] = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "est_prompt_tokens": 0,
            "errors": 0,
            "provider": self.provider,
            "model_name": self.model_name
        }
        self.audit_records: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_model_name(provider: str, model_name: str) -> str:
        """Normalizes model names to currently supported live endpoints."""
        if provider == "gemini":
            if not model_name or any(old in model_name for old in ["2.5", "2.0", "1.5", "1.0", "pro-vision"]):
                return "gemini-3.6-flash"
            return model_name
        elif provider == "openai":
            return model_name or "gpt-4o-mini"
        return model_name or "gemini-3.6-flash"

    def audit_all(
        self,
        train_dict: Dict[int, List[Tuple[int, int, int]]],
        R_RRFN: Dict[Tuple[int, int], float],
        ground_truth_labels: Optional[Dict[Tuple[int, int], int]] = None,
        user_mean_ratings: Optional[Dict[int, float]] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Audits all training interactions.
        - Statistically sound interactions (R_RRFN >= prefilter_threshold) get fast-path score = 0.95.
        - Suspicious interactions (R_RRFN < prefilter_threshold) are checked against SQLite cache.
        - For uncached items when provider != 'mock', up to max_live_calls are queried concurrently
          via thread pool with Gemini 3.6 Flash, and the rest are scored via heuristic mock.
        """
        R_LLM: Dict[Tuple[int, int], float] = {}
        gt_labels = ground_truth_labels or {}
        means = user_mean_ratings or {}

        # 1. Identify suspicious candidate interactions
        suspicious_list: List[Tuple[int, int, int]] = []
        for u, interactions in train_dict.items():
            for item, rating, ts in interactions:
                rrfn_score = R_RRFN.get((u, item), 0.5)
                if rrfn_score < self.prefilter_threshold:
                    suspicious_list.append((u, item, rating))
                else:
                    R_LLM[(u, item)] = 0.95  # Fast-path for statistically sound interactions

        # Sort suspicious by lowest RRFN score first (most anomalous items audited first)
        suspicious_list.sort(key=lambda x: R_RRFN.get((x[0], x[1]), 0.5))

        # 2. Check Cache & Separate Live vs Mock items
        to_query_live: List[Tuple[int, int, int, str, List[Tuple[int, int, int]]]] = []

        for u, item, rating in suspicious_list:
            cached = self.cache.get(u, item, rating, self.model_name)
            u_history = train_dict.get(u, [])
            profile_str = self.prompt_builder.build_user_profile_summary(u_history)
            prompt = self.prompt_builder.build_audit_prompt(u, profile_str, item, rating)
            title, genres = self.prompt_builder.item_info.get(item, (f"Movie_{item}", ["Unknown"]))
            gt = gt_labels.get((u, item), 1)

            if cached:
                score, reason = cached
                self.telemetry["cache_hits"] += 1
                R_LLM[(u, item)] = score
                if len(self.audit_records) < 30:
                    self.audit_records.append({
                        "user_id": u, "item_id": item, "title": title, "genres": genres,
                        "rating": rating, "score": score, "reason": reason, "prompt": prompt,
                        "is_cached": True, "source": self.provider, "ground_truth": gt
                    })
            elif self.provider != "mock" and len(to_query_live) < self.max_live_calls:
                to_query_live.append((u, item, rating, prompt, u_history))
            else:
                # Fast mock scoring for remaining items
                u_mean = means.get(u, 3.0)
                score, reason = self.mock.audit_interaction(u, item, rating, ground_truth_label=gt, user_mean_rating=u_mean)
                R_LLM[(u, item)] = score
                if len(self.audit_records) < 30:
                    self.audit_records.append({
                        "user_id": u, "item_id": item, "title": title, "genres": genres,
                        "rating": rating, "score": score, "reason": reason, "prompt": prompt,
                        "is_cached": False, "source": "mock" if self.provider == "mock" else f"{self.provider}-batch",
                        "ground_truth": gt
                    })

        # 3. Execute live queries concurrently
        if to_query_live:
            def _worker(item_tuple):
                u_id, itm_id, rat, p_text, hist = item_tuple
                try:
                    sc, reas = self._call_live_api(u_id, itm_id, rat, hist)
                    return (u_id, itm_id, rat, p_text, sc, reas, None)
                except Exception as ex:
                    return (u_id, itm_id, rat, p_text, self.fallback_score, f"API error: {str(ex)}", ex)

            # Concurrent execution with up to 5 worker threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_worker, t) for t in to_query_live]
                for future in concurrent.futures.as_completed(futures):
                    u_id, itm_id, rat, p_text, sc, reas, err = future.result()
                    title, genres = self.prompt_builder.item_info.get(itm_id, (f"Movie_{itm_id}", ["Unknown"]))
                    gt = gt_labels.get((u_id, itm_id), 1)

                    if err:
                        self.telemetry["errors"] += 1
                        u_mean = means.get(u_id, 3.0)
                        sc, reas = self.mock.audit_interaction(u_id, itm_id, rat, ground_truth_label=gt, user_mean_rating=u_mean)
                    else:
                        self.telemetry["api_calls"] += 1
                        self.telemetry["est_prompt_tokens"] += len(p_text.split()) * 2
                        self.cache.put(u_id, itm_id, rat, sc, reas, self.model_name)

                    R_LLM[(u_id, itm_id)] = sc
                    if len(self.audit_records) < 30:
                        self.audit_records.append({
                            "user_id": u_id, "item_id": itm_id, "title": title, "genres": genres,
                            "rating": rat, "score": sc, "reason": reas, "prompt": p_text,
                            "is_cached": False, "source": self.provider, "ground_truth": gt
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
        """Calls live LLM provider API with structured JSON output and fallback."""
        profile_str = self.prompt_builder.build_user_profile_summary(user_history)
        prompt = self.prompt_builder.build_audit_prompt(user_id, profile_str, item_id, rating)

        try:
            if self.provider == "gemini":
                try:
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=150,
                            response_mime_type="application/json"
                        )
                    )
                    return self._parse_json_response(response.text)
                except Exception as e_genai:
                    # Fallback to legacy SDK if google.genai has client issues
                    try:
                        import google.generativeai as legacy_genai
                        if self.api_key:
                            legacy_genai.configure(api_key=self.api_key)
                        model = legacy_genai.GenerativeModel(self.model_name)
                        response = model.generate_content(prompt)
                        return self._parse_json_response(response.text)
                    except Exception:
                        raise e_genai

            elif self.provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                return self._parse_json_response(response.choices[0].message.content)
            else:
                return self.fallback_score, f"Unsupported provider: {self.provider}"
        except Exception as e:
            self.telemetry["errors"] += 1
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                reason = "Gemini API quota exceeded (429); heuristic fallback applied."
            elif "404" in err_str or "NOT_FOUND" in err_str:
                reason = f"Gemini model {self.model_name} unavailable; heuristic fallback applied."
            else:
                reason = f"Live API Notice: {err_str[:80]}"
            return self.fallback_score, reason

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
