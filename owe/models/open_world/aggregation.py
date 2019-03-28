from typing import Tuple

import torch

from owe.utils import list_cat
from owe.config import Config


class Encoder(torch.torch.nn.Module):
    def __init__(self, embedding_m):
        super().__init__()
        self.embedding = torch.nn.Embedding(*embedding_m.shape, padding_idx=0)  # TODO remove hardcoded padding index
        self.embedding.weight.data.copy_(embedding_m)
        self.embedding.weight.require_grad = False
        self.delim_desc_start = [torch.tensor([2]).to(Config.get(Config.get("device")))]  # TODO remove hardcoded index
        self.delim_desc_end = [torch.tensor([3]).to(Config.get("device"))]  # TODO remove hardcoded index

    @staticmethod
    def pad(*args, **kwargs):
        """
        pads multiple lists of tensors. kwargs will be passed to torch pad_sequence # FIXME nothing is passed?
        :param args:
        :param kwargs:
        :return: Padded args
        """
        res = tuple(torch.nn.utils.rnn.pad_sequence(x, batch_first=True) for x in args)
        return res[0] if len(res) == 1 else res

    def forward(self, name: torch.tensor, description: torch.tensor):
        """
        :param name: List of word indices of words in name [B, ?]
        :param description: List of word indices in description [B, ??]
        :return: [B, d_emb]
        """
        raise NotImplementedError


class CNNEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor, output_d: int, n_filters: int = 3,
                 filter_sizes: Tuple[int, ...] = (2, 3, 5)):
        super().__init__(embedding_m)

        # TODO Add more parallel convs, concat and fc
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(1, output_d, (filter_sizes[k],
                                                                       self.embedding.weight.size(-1))).to(
                                                                            Config.get("device")
                                                                            )
                                         for k in range(n_filters)])
        # self.conv1 = torch.nn.Conv2d(in_channels=1,
        #                       out_channels=output_d,
        #                       kernel_size=(1, self.embedding.weight.size(-1)))
        # self.relu = torch.nn.ReLU()
        # torch.torch.nn.init.xavier_uniform_(self.conv1.weight)
        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, name: torch.tensor, description: torch.tensor) -> torch.tensor:
        seq = self.pad(list_cat((name, description), dim=-1))
        enc_seq = self.embedding(seq).unsqueeze(1)  # [B, 1, W, D]

        out = [k(enc_seq).squeeze(3) for k in self.conv]
        out = [torch.nn.functional.F.max_pool1d(k, k.size(2)).squeeze(2) for k in out]
        out = torch.cat(out, 1) if len(out) > 1 else out[0]
        # out = self.relu(out)
        # out = self.dropout(out)
        #
        # out = self.conv1(enc_seq) # [B, out_channels, kernel_size[0] over W,1]
        # out = out.squeeze(3) # [B, out_channels, kernel_size[0] over W]
        # out = F.max_pool1d(out, out.size(2)) #[B, out_channels, 1]
        # out = out.squeeze(2) #[B, out_channels]
        # out = self.relu(out)
        # out = self.dropout(self.relu(out)) # Also try with this removed
        return out


class LSTMEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor, output_d: int):
        super().__init__(embedding_m)
        self.bidirectional = Config.get("LSTMBidirectional")
        self.hidden_d = output_d // 2 if self.bidirectional else output_d  # TODO make configurable

        self.rnn = torch.nn.LSTM(
            input_size=self.embedding.weight.size(-1),
            hidden_size=self.hidden_d,
            num_layers=1,  # TODO we could change that
            dropout=0.3,  # TODO make configurable -> useless with num_layers <= 1.
            batch_first=True,
            bidirectional=self.bidirectional  # TODO we could change that
        )
        # self.word_dropout = torch.nn.Dropout2d(0.3)

    def forward(self, name: torch.tensor, description: torch.tensor):
        delim_desc_start = self.delim_desc_start * len(name)  # TODO fix hardcoded delimiter index
        delim_desc_end = self.delim_desc_end * len(name)  # TODO fix hardcoded delimiter index

        # seq = list_cat((name, delim_desc_start, description, delim_desc_end), dim=-1)        # [B, S1+S2]
        seq = list_cat((delim_desc_start, name, delim_desc_start, description, delim_desc_start), dim=-1)  # [B, S1+S2]
        seq = self.pad(seq)
        # from IPython import embed; embed()
        enc_seq = self.embedding(seq)  # [B, S1+S2, D]
        output, (_, _) = self.rnn(enc_seq)
        weighted_states = output[:, -1, :].squeeze()  # [B, H]
        return weighted_states


class AvgEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor):
        super().__init__(embedding_m)
        self.dropout = None
        if Config.get("AverageWordDropout"):
            self.dropout = torch.nn.Dropout2d(p=Config.get("AverageWordDropout"))

    def forward(self, name: torch.tensor, description: torch.tensor):
        """
        :param name: List of tensors [B, ?]
        :param description:
        :return: mean of embeddings in name+description
        """
        name, description = self.pad(name, description)  # [B, S1+S2]
        seq = torch.cat((name, description), dim=-1)  # [B, ?]
        enc_seq = self.embedding(seq)  # [B, S1+S2, D]
        if self.dropout:
            enc_seq = self.dropout(enc_seq)
        # Return mean during training and mean of non-zero rows during evaluation.
        # The noise caused by averaging zero rows seems to be beneficial in some cases.
        if self.training:
            return enc_seq.mean(dim=1)
        else:
            return torch.t(torch.t(enc_seq.sum(dim=1)).div((seq != 0).sum(dim=1).float())) # [B, D]
