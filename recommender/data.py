"""
data.py — MovieLens 100K loader and preprocessing.

Key design choices:
  - We keep it as (user, item, rating) triples throughout.
  - Train/test split is per-user (leave-one-out): the last interaction per user
    goes to the test set. This is the standard protocol for top-K evaluation.
  - We build dense integer indices for users and items (required by embedding layers).
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_movielens(data_dir: str = DATA_DIR) -> str:
    """Download and extract MovieLens 100K if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_path = os.path.join(data_dir, "ml-100k")

    if not os.path.exists(extract_path):
        print("Downloading MovieLens 100K ...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        print("Done.")
    return extract_path


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load u.data (100K ratings).
    Returns a DataFrame with columns: user_id, item_id, rating, timestamp.
    user_id and item_id are re-indexed to [0, N) for embedding layers.
    """
    path = download_movielens(data_dir)
    ratings_file = os.path.join(path, "u.data")

    df = pd.read_csv(
        ratings_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    # Re-index to 0-based contiguous integers
    user_map = {v: i for i, v in enumerate(df["user_id"].unique())}
    item_map = {v: i for i, v in enumerate(df["item_id"].unique())}
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map


def load_item_features(data_dir: str = DATA_DIR, item_map: dict = None) -> np.ndarray:
    """
    Load u.item and extract a feature vector per movie for content-based filtering.

    Feature vector (per movie):
      - release_year (normalized): 1 dim
      - 18 binary genre flags (Action, Adventure, Animation, ...)
    Total: 19 dimensions.
    """
    path = download_movielens(data_dir)
    item_file = os.path.join(path, "u.item")

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Childrens",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
        "SciFi", "Thriller", "War", "Western",
    ]
    cols = ["item_id", "title", "release_date", "video_date", "imdb_url"] + genre_cols

    items = pd.read_csv(
        item_file, sep="|", names=cols, encoding="latin-1", on_bad_lines="skip"
    )

    # Extract year from release_date (e.g., "01-Jan-1995" → 1995)
    items["year"] = pd.to_datetime(
        items["release_date"], errors="coerce"
    ).dt.year.fillna(1995)

    # Normalize year to [0, 1]
    year_min, year_max = items["year"].min(), items["year"].max()
    items["year_norm"] = (items["year"] - year_min) / max(year_max - year_min, 1)

    # Build feature matrix indexed by item_idx
    if item_map is None:
        raise ValueError("item_map is required to align features with embeddings.")

    n_items = len(item_map)
    n_features = 1 + len(genre_cols)  # year + genres
    feature_matrix = np.zeros((n_items, n_features), dtype=np.float32)

    for _, row in items.iterrows():
        orig_id = row["item_id"]
        if orig_id not in item_map:
            continue
        idx = item_map[orig_id]
        feature_matrix[idx, 0] = row["year_norm"]
        feature_matrix[idx, 1:] = row[genre_cols].values.astype(np.float32)

    return feature_matrix  # shape: (n_items, 19)


def train_test_split_leave_one_out(df: pd.DataFrame):
    """
    Leave-one-out split: for each user, the most recent interaction is the test item.
    Train: all interactions except the last per user.
    Test: one (user, item, rating) per user.
    """
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    test_mask = df_sorted.groupby("user_idx")["timestamp"].rank(
        method="first", ascending=False
    ) == 1
    train = df_sorted[~test_mask].reset_index(drop=True)
    test = df_sorted[test_mask].reset_index(drop=True)
    return train, test


class RatingsDataset(Dataset):
    """
    Simple (user, item, rating) dataset for matrix factorization.
    Ratings are normalized to [0, 1] so BCE loss can be used optionally.
    """

    def __init__(self, df: pd.DataFrame, normalize: bool = True):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        ratings = df["rating"].values.astype(np.float32)
        if normalize:
            ratings = (ratings - 1.0) / 4.0  # [1,5] → [0,1]
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class ContentDataset(Dataset):
    """
    Dataset for content-based filtering.
    Each sample: (user_idx, item_features, rating).
    User features: mean of all item features the user has rated (user profile vector).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_features: np.ndarray,
        normalize: bool = True,
    ):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_feats = torch.tensor(
            item_features[df["item_idx"].values], dtype=torch.float32
        )
        ratings = df["rating"].values.astype(np.float32)
        if normalize:
            ratings = (ratings - 1.0) / 4.0
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

        # Build user profile: mean of item features for items each user has rated
        n_users = df["user_idx"].max() + 1
        n_features = item_features.shape[1]
        user_profiles = np.zeros((n_users, n_features), dtype=np.float32)
        user_counts = np.zeros(n_users, dtype=np.float32)

        for row in df.itertuples():
            user_profiles[row.user_idx] += item_features[row.item_idx]
            user_counts[row.user_idx] += 1

        user_counts = np.maximum(user_counts, 1)
        user_profiles = user_profiles / user_counts[:, None]
        self.user_profiles = torch.tensor(
            user_profiles[df["user_idx"].values], dtype=torch.float32
        )

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # Input: concat(user_profile, item_features)
        x = torch.cat([self.user_profiles[idx], self.item_feats[idx]])
        return x, self.ratings[idx]


def get_data_loaders(
    data_dir: str = DATA_DIR, batch_size: int = 512
):
    """
    Full pipeline: download → load → split → return DataLoaders + metadata.
    """
    df, user_map, item_map = load_ratings(data_dir)
    item_features = load_item_features(data_dir, item_map)
    train_df, test_df = train_test_split_leave_one_out(df)

    n_users = len(user_map)
    n_items = len(item_map)

    # CF datasets
    cf_train = RatingsDataset(train_df)
    cf_test = RatingsDataset(test_df)

    # CBF datasets
    cbf_train = ContentDataset(train_df, item_features)
    cbf_test = ContentDataset(test_df, item_features)

    cf_train_loader = DataLoader(cf_train, batch_size=batch_size, shuffle=True)
    cf_test_loader = DataLoader(cf_test, batch_size=batch_size, shuffle=False)
    cbf_train_loader = DataLoader(cbf_train, batch_size=batch_size, shuffle=True)
    cbf_test_loader = DataLoader(cbf_test, batch_size=batch_size, shuffle=False)

    return {
        "cf": (cf_train_loader, cf_test_loader),
        "cbf": (cbf_train_loader, cbf_test_loader),
        "meta": {
            "n_users": n_users,
            "n_items": n_items,
            "n_features": item_features.shape[1],
            "train_df": train_df,
            "test_df": test_df,
            "item_features": item_features,
        },
    }