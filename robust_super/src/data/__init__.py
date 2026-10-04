from .loader import MovieLensLoader
from .amazon_loader import AmazonReviewLoader
from .yelp_loader import YelpLoader
from .preprocessor import DataSplit, preprocess_dataset, safe_join_path
from .attack_simulator import AttackSimulator

DATASET_LOADERS = {
    "movielens": MovieLensLoader,
    "amazon": AmazonReviewLoader,
    "amazon_electronics": AmazonReviewLoader,
    "yelp": YelpLoader,
}

def get_dataset_loader(dataset_name: str, data_dir: str, **kwargs):
    """
    Factory function returning the corresponding data loader instance.
    """
    key = dataset_name.lower().strip()
    if key not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[key](data_dir, **kwargs)

__all__ = [
    "MovieLensLoader",
    "AmazonReviewLoader",
    "YelpLoader",
    "DataSplit",
    "preprocess_dataset",
    "safe_join_path",
    "AttackSimulator",
    "DATASET_LOADERS",
    "get_dataset_loader"
]
