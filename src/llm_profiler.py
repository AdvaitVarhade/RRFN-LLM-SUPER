import json
import requests
import pandas as pd
from typing import List, Dict

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # or llama3.2:1b depending on system capacity

# System persona for user profiling
SYSTEM_PERSONA = (
    "You are an expert user profiler for a recommendation system. "
    "Your goal is to extract key preferences, genres, and sentiments from user interactions. "
    "Be concise and output ONLY a JSON object representing the user's top preferences."
)

def query_ollama(prompt: str) -> str:
    """
    Query the local Ollama instance with a prompt.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "system": SYSTEM_PERSONA,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "{}")
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return "{}"

def build_prompt_for_user(user_id: int, user_ratings: pd.DataFrame, movies_df: pd.DataFrame) -> str:
    """
    Construct a prompt based on the user's top-rated movies to extract preferences.
    """
    # Get top 5 highly rated movies by this user
    top_ratings = user_ratings[user_ratings['rating'] >= 4].sort_values(by='rating', ascending=False).head(5)
    
    if top_ratings.empty:
        return f"User {user_id} has no high ratings. Describe a generic user profile."
        
    movie_info = []
    for _, row in top_ratings.iterrows():
        movie = movies_df[movies_df['movie_id'] == row['movie_id']]
        if not movie.empty:
            title = movie['title'].values[0]
            genres = movie['genres'].values[0]
            movie_info.append(f"Title: {title}, Genres: {genres}")
            
    prompt = f"User {user_id} loved the following movies:\n"
    prompt += "\n".join(movie_info)
    prompt += "\nExtract the user's top 3 preferred genres and a short summary of their taste as JSON format with keys 'preferred_genres' (list) and 'taste_summary' (string)."
    return prompt

def generate_profiles_for_users(user_ids: List[int], ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Dict[int, dict]:
    """
    Generate LLM profiles for a batch of users.
    """
    profiles = {}
    for uid in user_ids:
        print(f"Generating profile for user {uid}...")
        prompt = build_prompt_for_user(uid, ratings_df[ratings_df['user_id'] == uid], movies_df)
        response_json = query_ollama(prompt)
        try:
            profile_data = json.loads(response_json)
            profiles[uid] = profile_data
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for user {uid}: {response_json}")
            profiles[uid] = {}
            
    return profiles

if __name__ == "__main__":
    from data_loader import load_data
    print("Loading data for testing LLM Profiler...")
    _, movies, ratings = load_data()
    
    # Test on a single user
    test_user_id = ratings['user_id'].iloc[0]
    print(f"Testing on User {test_user_id}")
    profile = generate_profiles_for_users([test_user_id], ratings, movies)
    print(f"Generated Profile: {json.dumps(profile, indent=2)}")
