import torch
import torch.nn as nn

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class ServiceGuardGAT(nn.Module):

    def __init__(
        self,
        num_classes,
        hidden_channels=64,
        heads=8
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels=4,
            out_channels=hidden_channels,
            heads=heads
        )

        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads
        )

        self.classifier = nn.Linear(
            hidden_channels * heads,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.gat1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.gat2(
            x,
            edge_index
        )

        x = global_mean_pool(
            x,
            batch
        )

        x = self.classifier(
            x
        )

        return x

        