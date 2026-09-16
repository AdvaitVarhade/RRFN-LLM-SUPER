import pandas as pd
from typing import Dict, List, Tuple, Optional

class LLMPromptBuilder:
    """
    Builds concise, structured prompts for the lightweight LLM semantic auditor.
    """
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df
        self.item_info: Dict[int, Tuple[str, List[str]]] = {}
        for _, row in movies_df.iterrows():
            item_id = int(row["item_id"])
            title = str(row.get("title", f"Movie_{item_id}"))
            genres = row.get("genres", ["Unknown"])
            if isinstance(genres, str):
                genres = [genres]
            self.item_info[item_id] = (title, genres)

    def build_user_profile_summary(self, user_interactions: List[Tuple[int, int, int]]) -> str:
        """
        Builds a compact summary of user preferences from historical (item, rating, ts) tuples.
        """
        if not user_interactions:
            return "User has no prior interaction history."

        genre_ratings: Dict[str, List[int]] = {}
        top_titles: List[str] = []

        # Sort by rating descending
        sorted_history = sorted(user_interactions, key=lambda x: -x[1])

        for item_id, rating, _ in sorted_history:
            title, genres = self.item_info.get(item_id, (f"Movie_{item_id}", ["Unknown"]))
            if rating >= 4 and len(top_titles) < 4:
                top_titles.append(f"{title} ({rating}/5)")
            for g in genres:
                genre_ratings.setdefault(g, []).append(rating)

        # Compute average rating per genre
        genre_means = {g: sum(r) / len(r) for g, r in genre_ratings.items() if len(r) >= 2}
        liked_genres = [g for g, m in sorted(genre_means.items(), key=lambda x: -x[1]) if m >= 3.5][:3]
        disliked_genres = [g for g, m in sorted(genre_means.items(), key=lambda x: x[1]) if m <= 2.5][:2]

        liked_str = ", ".join(liked_genres) if liked_genres else "General Cinema"
        disliked_str = ", ".join(disliked_genres) if disliked_genres else "None noted"
        top_str = "; ".join(top_titles) if top_titles else "Various titles"

        return f"Preferred Genres: [{liked_str}]. Disliked Genres: [{disliked_str}]. Highly Rated: [{top_str}]."

    def build_audit_prompt(
        self,
        user_id: int,
        user_profile_summary: str,
        item_id: int,
        observed_rating: int,
        review_text: Optional[str] = None
    ) -> str:
        """
        Creates an audit prompt requesting a semantic reliability score between 0.0 and 1.0 in JSON format.
        """
        title, genres = self.item_info.get(item_id, (f"Movie_{item_id}", ["Unknown"]))
        genre_str = ", ".join(genres)

        review_section = f'\nReview Comment: "{review_text}"' if review_text else ""

        prompt = f"""System: You are an expert Recommendation Data Integrity Auditor.
Evaluate whether the recorded rating is semantically consistent with the user's documented taste profile.

User ID: {user_id}
User Profile: {user_profile_summary}
Evaluated Item: "{title}"
Item Genres: [{genre_str}]
Recorded Rating: {observed_rating} Stars (1 to 5 scale){review_section}

Task: Determine if this rating represents authentic user preference or anomalous noise (e.g. rating flip, review bombing, or spam).
Respond strictly in JSON format with two keys:
{{
  "semantic_reliability": <float from 0.0 (highly suspicious / contradictory) to 1.0 (highly authentic / coherent)>,
  "reason": "<one sentence justification>"
}}
"""
        return prompt
