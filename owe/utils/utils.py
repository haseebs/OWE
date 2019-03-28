import configparser
import logging
import pathlib as pl
import shutil
from typing import Any, Dict, Optional

import torch
import numpy as np

from owe.config import Config

logger = logging.getLogger("owe")


def list_cat(args, dim=-1):
    """
    Concatenates the tensors in multiple lists of tensors along the given dimension.
    It returns another list of tensors.
    :param args: a tuple of lists of tensors (tensors will be concatenated)
    :param dim:
    :return:
    """
    return [torch.cat(x, dim=dim) for x in zip(*args)]


def read_config(config_file):
    """
    Reads the config file.
    :param config_file: Path to config file
    :return: train_file, val_file, test_file, skip_header, split_symbol,
             entity2wiki_file
    """
    config = configparser.ConfigParser()
    config.read(str(config_file))
    logger.info("Reading config from: %s" % config_file)

    # GPUs
    Config.set("DisableCuda", config.getboolean("GPU", "DisableCuda", fallback=False))
    if not Config.get("DisableCuda") and torch.cuda.is_available():
            Config.set("device", 'cuda')
    else:
            Config.set("device", 'cpu')
    Config.set("GPUs", [int(gpu) for gpu in config.get("GPU", "GPUs", fallback='').split(',')])

    # Training
    # Config.set("LinkPredictionModelType", config.get("Training", "LinkPredictionModelType", fallback=None))
    Config.set("Epochs", config.getint("Training", "Epochs", fallback=1000))
    Config.set("BatchSize", config.getint("Training", "BatchSize", fallback=None))
    Config.set("EmbeddingDimensionality", config.getint("Training", "EmbeddingDimensionality", fallback=300))
    Config.set("LearningRate", config.getfloat("Training", "LearningRate", fallback=0.1))
    Config.set("LearningRateSchedule", config.get("Training", "LearningRateSchedule", fallback="50,100,200").split(','))
    Config.set("LearningRateGammas", config.get("Training", "LearningRateGammas", fallback="0.1,0.1,0.1").split(','))
    Config.set("InitializeEmbeddingWithAllEntities",
               config.getboolean("Training", "InitializeEmbeddingWithAllEntities", fallback=False))
    Config.set("InitializeWithPretrainedKGCEmbedding",
               config.getboolean("Training", "InitializeWithPretrainedKGCEmbedding", fallback=False))

    Config.set("TransformationType", config.get("Training", "TransformationType", fallback="Linear"))
    Config.set("EncoderType", config.get("Training", "EncoderType", fallback="Average"))
    Config.set("UseTailsToOptimize", config.getboolean("Training", "UseTailsToOptimize", fallback=False))
    Config.set("Loss", config.get("Training", "Loss", fallback="Pairwise"))
    Config.set("UNKType", config.get("Training", "UNKType", fallback="Average"))
    Config.set("AverageWordDropout", config.getfloat("Training", "AverageWordDropout", fallback=0.))
    Config.set("IterTriplets", config.getboolean("Training", "IterTriplets", fallback=True))

    # FCN
    Config.set("FCNUseSigmoid", config.getboolean("FCN", "FCNUseSigmoid", fallback=False))
    Config.set("FCNLayers", config.getint("FCN", "FCNLayers", fallback=0))
    Config.set("FCNDropout", config.getfloat("FCN", "FCNDropout", fallback=0))
    Config.set("FCNHiddenDim", config.getint("FCN", "FCNHiddenDim", fallback=None))

    # LSTM
    Config.set("LSTMOutputDim", config.getint("LSTM", "LSTMOutputDim", fallback=None))
    Config.set("LSTMBidirectional", config.getboolean("LSTM", "LSTMBidirectional", fallback=False))

    # Evaluation
    Config.set("ValidateEvery", config.getint("Evaluation", "ValidateEvery", fallback=1000))
    Config.set("UseTargetFilteringShi", config.getboolean("Evaluation", "UseTargetFilteringShi", fallback=False))
    Config.set("PrintTrainNN", config.getboolean("Evaluation", "PrintTrainNN", fallback=False))
    Config.set("PrintTestNN", config.getboolean("Evaluation", "PrintTestNN", fallback=False))
    Config.set("EvalRandomHeads", config.getboolean("Evaluation", "EvalRandomHeads", fallback=False))
    Config.set("CalculateNNMeanRank", config.getboolean("Evaluation", "CalculateNNMeanRank", fallback=False))
    Config.set("ShiTargetFilteringBaseline",
               config.getboolean("Evaluation", "ShiTargetFilteringBaseline", fallback=False))
    Config.set("GetTensorboardEmbeddings", config.getboolean("Evaluation", "GetTensorboardEmbeddings", fallback=True))

    if not len(Config.get("LearningRateSchedule")) == len(Config.get("LearningRateGammas")):
        raise ValueError("Length of LearningRateSchedule must be equal to LearningRateGammas")

    # early stopping
    Config.set("EarlyStopping", config.getboolean("EarlyStopping", "EarlyStopping", fallback=False))
    Config.set("EarlyStoppingThreshold", config.getfloat("EarlyStopping", "EarlyStoppingThreshold", fallback=0.1))
    Config.set("EarlyStoppingLastX", config.getint("EarlyStopping", "EarlyStoppingLastX", fallback=10))
    Config.set("EarlyStoppingMinEpochs", config.getint("EarlyStopping", "EarlyStoppingMinEpochs", fallback=10))

    # Entity2text
    Config.set("PretrainedEmbeddingFile", config.get("Entity2Text", "PretrainedEmbeddingFile", fallback=None))
    Config.set("ConvertEntities", config.getboolean("Entity2Text", "ConvertEntities", fallback=False))
    Config.set("ConvertEntitiesWithMultiprocessing",
               config.getboolean("Entity2Text", "ConvertEntitiesWithMultiprocessing", fallback=True))
    Config.set("MatchTokenInEmbedding", config.getboolean("Entity2Text", "MatchTokenInEmbedding", fallback=False))
    Config.set("MatchLabelInEmbedding", config.getboolean("Entity2Text", "MatchLabelInEmbedding", fallback=False))
    Config.set("LimitDescription", config.getint("Entity2Text", "LimitDescription", fallback=100000))

    # logger.info("LinkPredictionModelType: %s " % Config.get("LinkPredictionModelType"))
    # if Config.get("LinkPredictionModelType") not in ["ComplEx", "TransE", "TransR", "DistMult"]:
    #     raise ValueError("LinkPredictionModelType not recognized")

    # Dataset
    train_file = config.get("Dataset", "TrainFile")
    valid_file = config.get("Dataset", "ValidationFile")
    test_file = config.get("Dataset", "TestFile")
    entity2wiki_file = config.get("Dataset", "Entity2wikidata", fallback="entity2wikidata.json")
    logger.info("Using {} as wikidata file".format(entity2wiki_file))
    skip_header = config.getboolean("Dataset", "SkipHeader", fallback=False)
    split_symbol = config.get("Dataset", "SplitSymbol", fallback='TAB')
    if split_symbol not in ["TAB", "SPACE"]:
        raise ValueError("SplitSymbol must be either TAB or SPACE.")
    split_symbol = '\t' if split_symbol == 'TAB' else ' '
    return train_file, valid_file, test_file, skip_header, split_symbol, entity2wiki_file


def load_checkpoint(checkpoint_file: pl.Path) -> Optional[Dict[str, Any]]:
    """
    Loads the OWE checkpoint (not the KGC embeddings)
    :param checkpoint_file: file name for the latest checkpoint file
    """
    if checkpoint_file.exists():
        logger.info(f"Loading checkpoint {checkpoint_file}.")
        checkpoint = torch.load(str(checkpoint_file))
        logger.info(f"Done loading checkpoint from epoch {checkpoint['epoch']}.")
    else:
        logger.warning(f"No {checkpoint_file} checkpoint file found. Starting normal.")
    return checkpoint


def save_checkpoint(epoch, outdir, model, mapper, optimizer, criterion,
                    filename='checkpoint.OWE.pth.tar'):
    """
    Saves the OWE checkpoint (not the KGC embeddings)
    :param epoch: current epoch
    :param outdir: directory to output checkpoint
    :param model: owe model to save
    :param mapper: mapper to save
    :param optimizer: optimizer for the model
    :param criterion: criterion to use for determining best ckpt (list)
    :param filename: for checkpoint in outdir
    :return:
    """
    filename = outdir / filename
    logger.info("Saving checkpoint to {}.".format(filename))
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'mapper': mapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, str(filename))
    if max(criterion) == criterion[-1]:
        best_name = str(outdir / 'best_checkpoint.OWE.pth.tar')
        shutil.copyfile(str(filename), best_name)
        logger.info("Saved best checkpoint to {}.".format(best_name))


def decay_lr(current_epoch, optimizer):
    """
    Decays LR according to the schedule and gammas given in the config
    :param current_epoch:
    :param optimizer:
    :return:
    """
    schedule = Config.get("LearningRateSchedule")
    gammas = Config.get("LearningRateGammas")
    for idx, target_epoch in enumerate(schedule):
        if current_epoch == int(target_epoch):
            for params in optimizer.param_groups:
                current_lr = params['lr']
                params['lr'] = current_lr * float(gammas[idx])
                logger.info("Changed lr from {} to {}".format(current_lr,
                                                              params['lr']))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cosine_loss(x, y):
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=1)
