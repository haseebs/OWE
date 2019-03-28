import logging

import torch

from owe.models.open_world.aggregation import AvgEncoder, CNNEncoder, LSTMEncoder
from owe.models.open_world.transformation import LinearTransform, FCNTransform
from owe.config import Config

logger = logging.getLogger('owe')


class Mapper(torch.nn.Module):
    """
    Maps sequences of words to a KGC embedding to be used with
    a pretrained KGC model.

    """
    def __init__(self, embedding_m: torch.tensor, complex_d: int):
        super().__init__()
        encoder_out_d = embedding_m.size(1)

        if Config.get("EncoderType") == "Average":
            self.encoder = AvgEncoder(embedding_m)
            logger.info("Using averaging encoder")
        elif Config.get("EncoderType") == "CNN":
            self.encoder = CNNEncoder(embedding_m, output_d=encoder_out_d)
            logger.info("Using CNN encoder")
        elif Config.get("EncoderType") == "BiLSTM":
            if Config.get("LSTMOutputDim"):
                encoder_out_d = Config.get("LSTMOutputDim")
            self.encoder = LSTMEncoder(embedding_m, output_d=encoder_out_d)
            logger.info("Using BiLSTM encoder")
        else:
            raise ValueError("EncoderType invalid in config")

        if Config.get("LinkPredictionModelType") in ("ComplEx", "RotatE"):
            if Config.get("TransformationType") == "Linear":
                self.transformer_r = LinearTransform(encoder_out_d, complex_d)
                self.transformer_i = LinearTransform(encoder_out_d, complex_d)
                logger.info("Using Linear transformation")
            elif Config.get("TransformationType") == "Affine":
                self.transformer_r = LinearTransform(encoder_out_d, complex_d, bias=True)
                self.transformer_i = LinearTransform(encoder_out_d, complex_d, bias=True)
                logger.info("Using Affine transformation")
            elif Config.get("TransformationType") == "FCN":
                use_sigmoid = Config.get("FCNUseSigmoid")
                n_layers = Config.get("FCNLayers")

                self.transformer_r = FCNTransform(encoder_out_d, complex_d,
                                                  hidden_dim=Config.get("FCNHiddenDim"),
                                                  n_layers=n_layers,
                                                  use_sigmoid=use_sigmoid)
                self.transformer_i = FCNTransform(encoder_out_d, complex_d,
                                                  hidden_dim=Config.get("FCNHiddenDim"),
                                                  n_layers=n_layers,
                                                  use_sigmoid=use_sigmoid)
                logger.info("Using FCN transformation")
            else:
                raise ValueError("TransformationType invalid in config")
        elif Config.get("LinkPredictionModelType") in ("TransE", "TransR", "DistMult"):
            if Config.get("TransformationType") == "Linear":
                self.transformer = LinearTransform(encoder_out_d, complex_d)
                logger.info("Using Linear transformation")
            elif Config.get("TransformationType") == "Affine":
                self.transformer = LinearTransform(encoder_out_d, complex_d, bias=True)
                logger.info("Using Affine transformation")
            elif Config.get("TransformationType") == "FCN":
                use_sigmoid = Config.get("FCNUseSigmoid")
                n_layers = Config.get("FCNLayers")

                self.transformer = FCNTransform(encoder_out_d, complex_d,
                                                hidden_dim=Config.get("FCNHiddenDim"),
                                                n_layers=n_layers,
                                                use_sigmoid=use_sigmoid)
                logger.info("Using FCN transformation")
            else:
                raise ValueError("TransformationType invalid in config")
        else:
            raise ValueError("LinkPredictionModelType unknown")

    def forward(self, name: torch.tensor, description: torch.tensor):
        enc = self.encoder(name, description)

        if Config.get("LinkPredictionModelType") in ("ComplEx", "RotatE"):
            return self.transformer_r(enc), self.transformer_i(enc)
        else:
            return self.transformer(enc)
