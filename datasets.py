import os
import urllib
import zipfile

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(__file__, ".."))


class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, n_negative):
        super().__init__()
        print("Loading movielens dataset...")
        df = self.__load_movielens_ratings()
        user_ids = df["userId"].drop_duplicates().reset_index(drop=True)
        movie_ids = df["movieId"].drop_duplicates().reset_index(drop=True)
        user_id_to_index = {user_id: index for index, user_id in user_ids.iteritems()}
        movie_id_to_index = {movie_id: index for index, movie_id in movie_ids.iteritems()}

        self.n_users = len(user_ids)
        self.n_items = len(movie_ids)

        print("Constructing matrix...")
        matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        interactions = []
        for user_id, movie_id, rating, timestamp in df.values:
            u = user_id_to_index[user_id]
            i = movie_id_to_index[movie_id]
            matrix[u, i] = 1.0
            interactions.append((u, i))

        print("Creating training data...")
        self.user_inputs = []
        self.item_inputs = []
        self.labels = []

        for u, i in tqdm(interactions):
            if matrix[u, i] == 1.0:
                self.user_inputs.append(u)
                self.item_inputs.append(i)
                self.labels.append(1)

                for _ in range(n_negative):
                    j = np.random.randint(self.n_items)
                    while matrix[u, j] != 0.0:
                        j = np.random.randint(self.n_items)
                    self.user_inputs.append(u)
                    self.item_inputs.append(j)
                    self.labels.append(0)

    def __len__(self):
        return len(self.user_inputs)

    def __getitem__(self, index):
        user_input = self.user_inputs[index]
        item_input = self.item_inputs[index]
        label = self.labels[index]
        return (
            torch.tensor(user_input, dtype=torch.long),
            torch.tensor(item_input, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )

    def __load_movielens_ratings(self):
        data_dir = os.path.join(BASE_DIR, "data")
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        filepath = os.path.join(data_dir, "movielens.zip")
        if not os.path.exists(filepath):
            url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            urllib.request.urlretrieve(url, filepath)
            zip_ref = zipfile.ZipFile(filepath, "r")
            zip_ref.extractall(data_dir)

        df = pd.read_csv(os.path.join(data_dir, "ml-latest-small", "ratings.csv"))
        return df
