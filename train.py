import argparse
import datetime
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


class GMF(torch.nn.Module):
    def __init__(self, n_users, n_items, latent_dim):
        super().__init__()
        self.user_embedding_layer = torch.nn.Embedding(n_users, latent_dim)
        self.item_embedding_layer = torch.nn.Embedding(n_items, latent_dim)
        self.h = torch.nn.Linear(latent_dim, 1)
        self.a = torch.nn.Sigmoid()

    def forward(self, user_vector, item_vector):
        user_embedding = self.user_embedding_layer(user_vector)
        item_embedding = self.item_embedding_layer(item_vector)
        mf = torch.mul(user_embedding, item_embedding)
        out = self.a(self.h(mf))
        return out


class MLP(torch.nn.Module):
    def __init__(self, n_users, n_items, latent_dim, n_mlp_layers):
        super().__init__()
        self.user_embedding_layer = torch.nn.Embedding(n_users, latent_dim)
        self.item_embedding_layer = torch.nn.Embedding(n_items, latent_dim)

        self.mlp_layers = torch.nn.ModuleList()
        units = latent_dim * 2
        for _ in range(n_mlp_layers):
            self.mlp_layers.append(torch.nn.Linear(units, units // 2))
            self.mlp_layers.append(torch.nn.ReLU())
            units //= 2

    def forward(self, user_vector, item_vector):
        user_embedding = self.user_embedding_layer(user_vector)
        item_embedding = self.item_embedding_layer(item_vector)
        concatenated = torch.cat([user_embedding, item_embedding], dim=1)
        out = concatenated
        for mlp_layer in self.mlp_layers:
            out = mlp_layer(out)
        return out


class NeuMF(torch.nn.Module):
    def __init__(self, n_users, n_items, gmf_latent_dim, mlp_latent_dim, n_mlp_layers):
        super().__init__()
        self.gmf = GMF(n_users, n_items, gmf_latent_dim)
        self.mlp = MLP(n_users, n_items, mlp_latent_dim, n_mlp_layers)
        mlp_out_dim = mlp_latent_dim // (2 ** (n_mlp_layers - 1))
        self.neumf_layer = torch.nn.Linear(mlp_out_dim + 1, 1)

    def forward(self, user_vector, item_vector):
        gmf_out = self.gmf(user_vector, item_vector)
        mlp_out = self.mlp(user_vector, item_vector)
        concatenated = torch.cat([gmf_out, mlp_out], dim=1)
        neumf = self.neumf_layer(concatenated)
        out = torch.sigmoid(neumf)
        return out


def train(args):
    dataset = MovieLensDataset(args.n_negative)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    neumf = NeuMF(
        dataset.n_users,
        dataset.n_items,
        args.gmf_latent_dim,
        args.mlp_latent_dim,
        args.n_mlp_layers,
    )

    if args.load_model_path:
        neumf.load_state_dict(torch.load(args.load_model_path))
    if torch.cuda.is_available():
        neumf = neumf.cuda()

    n_steps = len(dataset) // args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(neumf.parameters(), lr=args.learning_rate)

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        avg_loss = 0.0
        for step, (user_vector, item_vector, labels) in enumerate(data_loader, 0):
            optimizer.zero_grad()

            outputs = neumf(user_vector, item_vector).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss += loss.item()
            if step % 100 == 99:
                print(
                    "[epoch %d/%d, step %5d/%d] loss: %.3f"
                    % (epoch + 1, args.n_epochs, step + 1, n_steps, running_loss / 100)
                )
                running_loss = 0.0

        avg_loss /= n_steps
        print("epoch %d's average loss: %.3f\n" % (epoch + 1, avg_loss))

        model_name = "ncf_%s.pt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(args.save_dir, model_name)
        torch.save(neumf.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_negative", type=int, default=4, help="the number of negative instances"
    )
    parser.add_argument(
        "--gmf_latent_dim",
        type=int,
        default=32,
        help="the dimension of GMF latent vector",
    )
    parser.add_argument(
        "--mlp_latent_dim",
        type=int,
        default=32,
        help="the dimension of MLP latent vector",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument(
        "--n_mlp_layers", type=int, default=4, help="the number of MLP layers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="the number of epochs"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="path to new models' directory",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="path to existing model to be loaded",
    )
    train(parser.parse_args())
