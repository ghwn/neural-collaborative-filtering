import argparse
import datetime
import os

import torch

from datasets import MovieLensDataset
from models import NeuMF


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
