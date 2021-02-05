import torch


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
