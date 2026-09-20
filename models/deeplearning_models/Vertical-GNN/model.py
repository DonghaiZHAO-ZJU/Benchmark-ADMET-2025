import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class GIN(torch.nn.Module):

    def __init__(self, num_features, num_targets, num_layers, hidden_size):
        super(GIN, self).__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(
                    GINConv(
                        Sequential(
                            Linear(num_features, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
            else:
                layers.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
        self.model = nn.Sequential(*layers)
        # Readout MLP

        self.ro = Sequential(
            Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, num_targets)
        )

    def forward(self, x, edge_index, batch_index):
        for layer in self.model:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch_index)

        return self.ro(x)


# Build graph transformer model using TransformerConv from pytorch geometric.


class GraphTrans(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        hidden_size,
        n_heads,
        dropout,
        edge_dim,
    ):
        super(GraphTrans, self).__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(
                    TransformerConv(
                        num_features,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )
            else:
                layers.append(
                    TransformerConv(
                        hidden_size * n_heads,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )

        self.model = nn.Sequential(*layers)
        # Readout MLP
        self.ro = Sequential(
            Linear(hidden_size * n_heads, hidden_size),
            ReLU(),
            Linear(hidden_size, num_targets),
        )

    def forward(self, x, edge_attr, edge_index, batch_index):
        for layer in self.model:
            x = layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch_index)
        return self.ro(x)


class ParallelGNN(torch.nn.Module):
    """
    Training in parallel
    """

    def __init__(
        self,
        num_features,
        num_targets,
        num_gin_layers,
        num_graph_trans_layers,
        hidden_size,
        n_heads,
        dropout,
        edge_dim,
    ):
        super(ParallelGNN, self).__init__()
        gin_layers = []
        graph_trans_layers = []
        for _ in range(num_gin_layers):
            if len(gin_layers) == 0:
                gin_layers.append(
                    GINConv(
                        Sequential(
                            Linear(num_features, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
            else:
                gin_layers.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )

        for _ in range(num_graph_trans_layers):
            if len(graph_trans_layers) == 0:
                graph_trans_layers.append(
                    TransformerConv(
                        num_features,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )
            else:
                graph_trans_layers.append(
                    TransformerConv(
                        hidden_size * n_heads,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )

        self.gin_model = nn.Sequential(*gin_layers)
        self.graph_trans_model = nn.Sequential(*graph_trans_layers)
        # Readout MLP
        self.ro = Sequential(
            Linear(hidden_size + hidden_size * n_heads, hidden_size),
            ReLU(),
            Linear(hidden_size, num_targets),
        )

    def forward(self, x, edge_attr, edge_index, batch_index):
        all_embeds = []
        for layer_no, layer in enumerate(self.gin_model):
            if layer_no == 0:
                gin_x = layer(x, edge_index)
            else:
                gin_x = layer(gin_x, edge_index)

        gin_batch_embeds = global_add_pool(gin_x, batch_index)
        all_embeds.append(gin_batch_embeds)
        for layer_no, layer in enumerate(self.graph_trans_model):
            if layer_no == 0:
                gt_x = layer(x, edge_index, edge_attr)
            else:
                gt_x = layer(gt_x, edge_index, edge_attr)

        graph_trans_batch_embeds = global_mean_pool(gt_x, batch_index)
        all_embeds.append(graph_trans_batch_embeds)

        concat_embeddings = torch.cat(all_embeds, dim=1)
        # read out
        out = self.ro(concat_embeddings)
        return out


class VerticalGNN(torch.nn.Module):
    """
    Training in vertical
    """

    def __init__(
        self,
        num_features,
        num_targets,
        num_gin_layers,
        num_graph_trans_layers,
        hidden_size,
        n_heads,
        dropout,
        edge_dim,
    ):
        super(VerticalGNN, self).__init__()
        gin_layers = []
        graph_trans_layers = []

        for _ in range(num_graph_trans_layers):
            if len(graph_trans_layers) == 0:
                graph_trans_layers.append(
                    TransformerConv(
                        num_features,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )
            else:
                graph_trans_layers.append(
                    TransformerConv(
                        hidden_size * n_heads,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )

        for _ in range(num_gin_layers):
            if len(gin_layers) == 0:
                gin_layers.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_size * n_heads, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
            else:
                gin_layers.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )

        self.gin_model = nn.Sequential(*gin_layers)
        self.graph_trans_model = nn.Sequential(*graph_trans_layers)

        # Readout MLP
        self.ro = Sequential(
            Linear(hidden_size + hidden_size * n_heads, hidden_size),
            ReLU(),
            Linear(hidden_size, num_targets),
        )

    def forward(self, x, edge_attr, edge_index, batch_index):
        all_embeds = []

        for layer in self.graph_trans_model:
            x = layer(x, edge_index, edge_attr)

        gt_embeds = global_mean_pool(x, batch_index)
        all_embeds.append(gt_embeds)

        for layer in self.gin_model:
            x = layer(x, edge_index)

        gin_embeds = global_add_pool(x, batch_index)
        all_embeds.append(gin_embeds)

        gt_gin_embeds = torch.cat(all_embeds, dim=1)

        return self.ro(gt_gin_embeds)
