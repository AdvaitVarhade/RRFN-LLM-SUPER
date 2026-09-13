import os
import zipfile
import requests
import pandas as pd
from io import BytesIO
from tqdm import tqdm

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ML1M_DIR = os.path.join(DATA_DIR, "ml-1m")

def download_and_extract_movielens():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if os.path.exists(ML1M_DIR):
        print(f"MovieLens-1M already exists at {ML1M_DIR}")
        return

    print(f"Downloading MovieLens-1M from {DATA_URL}...")
    response = requests.get(DATA_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with BytesIO() as b:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="ml-1m.zip") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    b.write(chunk)
                    pbar.update(len(chunk))
        
        b.seek(0)
        print("Extracting...")
        with zipfile.ZipFile(b) as z:
            z.extractall(DATA_DIR)
            
    print("Download and extraction complete.")

def load_data():
    if not os.path.exists(ML1M_DIR):
        download_and_extract_movielens()
        
    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv(os.path.join(ML1M_DIR, 'users.dat'), sep='::', engine='python', names=users_cols, encoding='latin-1')
    
    movies_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(os.path.join(ML1M_DIR, 'movies.dat'), sep='::', engine='python', names=movies_cols, encoding='latin-1')
    
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(ML1M_DIR, 'ratings.dat'), sep='::', engine='python', names=ratings_cols, encoding='latin-1')
    
    return users, movies, ratings

def create_federated_partitions(ratings, num_clients=100):
    """
    Partition ratings data by user_id to simulate federated learning clients.
    """
    print(f"Partitioning data into {num_clients} clients...")
    user_ids = ratings['user_id'].unique()
    
    # In a real FL scenario, each user might be a client. 
    # Here we group users into num_clients buckets to limit simulation overhead.
    client_data = {}
    for i in range(num_clients):
        client_data[f"client_{i}"] = []
        
    for idx, user in enumerate(user_ids):
        client_id = f"client_{idx % num_clients}"
        user_ratings = ratings[ratings['user_id'] == user]
        client_data[client_id].append(user_ratings)
        
    for client in client_data:
        client_data[client] = pd.concat(client_data[client])
        
    return client_data

if __name__ == "__main__":
    users, movies, ratings = load_data()
    print(f"Loaded {len(users)} users, {len(movies)} movies, and {len(ratings)} ratings.")
    
    # Test partitioning
    clients = create_federated_partitions(ratings, num_clients=100)
    print(f"Successfully created {len(clients)} client partitions.")
