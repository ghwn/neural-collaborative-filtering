import os
import urllib
import zipfile

import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.abspath(os.path.join(__file__, ".."))


class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, n_negative):
        super().__init__()
        ratings = self.__load_movielens_ratings()
        ratings["userId"] = ratings["userId"] - 1
        ratings["movieId"] = ratings["movieId"] - 1
        user_ids = sorted(list(set(ratings["userId"])))
        item_ids = sorted(list(set(ratings["movieId"])))
        self.index_to_user = {i: user_id for i, user_id in enumerate(user_ids)}
        self.user_to_index = {user_id: i for i, user_id in enumerate(user_ids)}
        self.index_to_item = {i: item_id for i, item_id in enumerate(item_ids)}
        self.item_to_index = {item_id: i for i, item_id in enumerate(item_ids)}

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        matrix = np.zeros((self.n_users, self.n_items), dtype=np.uint8)
        for user_id, movie_id, rating, timestamp in ratings.values:
            matrix[int(user_id), self.item_to_index[int(movie_id)]] = 1

        self.user_inputs = []
        self.item_inputs = []
        self.labels = []
        for u in range(len(matrix)):
            for i in range(len(matrix[u])):
                if matrix[u, i] != 1:
                    continue

                self.user_inputs.append(u)
                self.item_inputs.append(i)
                self.labels.append(1)

                for _ in range(n_negative):
                    while True:
                        j = np.random.randint(0, self.n_items)
                        if matrix[u, j] == 0:
                            break
                    self.user_inputs.append(u)
                    self.item_inputs.append(j)
                    self.labels.append(0)

    def __len__(self):
        return len(self.user_inputs)

    def __getitem__(self, index):
        user_input = self.user_inputs[index]
        item_input = self.item_inputs[index]
        label = self.labels[index]
        tensors = (
            torch.tensor(user_input, dtype=torch.long),
            torch.tensor(item_input, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )
        if torch.cuda.is_available():
            return tuple([tensor.cuda() for tensor in tensors])
        else:
            return tensors

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
