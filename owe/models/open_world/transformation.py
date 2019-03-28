from typing import Optional

import torch

from owe.config import Config


class LinearTransform(torch.nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, bias: bool = False):
        """
        :param int inp_dim:
        :param int out_dim:
        """
        super(LinearTransform, self).__init__()
        self.mat = torch.nn.Linear(inp_dim, out_dim, bias=bias)

    def init_weight_ones(self):
        self.mat.weight.data.fill_(1)
        self.mat.bias.data.fill_(1)

    def forward(self, x: torch.tensor, y: Optional[torch.tensor] = None):
        """
        :param x: input embedding
        :param y: input embedding
        """
        if y is not None:
            return self.mat(x), self.mat(y)
        else:
            return self.mat(x)


class FCNTransform(torch.nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, n_layers: int = 0,
                 hidden_dim: int = None, use_sigmoid: bool = False, use_dropout: bool = False):
        """
        :param int inp_dim:
        :param int out_dim:
        :param int n_layers: number of intermediate layers
        :param int hidden_dim:
        :param int use_sigmoid: use sigmoid act if true else relu
        """
        super(FCNTransform, self).__init__()
        hidden_dim = hidden_dim or inp_dim
        if use_sigmoid:
            self.act = torch.nn.sigmoid()
        else:
            self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=Config.get("FCNDropout")) if use_dropout else None

        layers = [torch.nn.Linear(inp_dim, hidden_dim), self.act]
        for _ in range(n_layers):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.act)
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        self.layers = torch.nn.ModuleList(layers)
        print(self)

    def forward(self, x: torch.tensor):
        """
        :param x: input embeddings:
        """
        out = x
        for i, k in enumerate(self.layers):
            out = k(out)
            if i < (len(self.layers) - 1) and self.dropout:
                out = self.dropout(out)
        return out
