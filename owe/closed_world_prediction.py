import logging
from typing import Tuple, List

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("owe")


def compute_target_ranks(scores: torch.Tensor, targets: torch.Tensor,
                         labels: torch.Tensor = None, filtered: bool = True) -> List[float]:
    """
    Finds the rank of the target entity inside the scores tensor.

    :param scores: Scores tensor of shape [B, E] (batch_size, entities)
    :param targets: Target tensor of shape [B]. Can be either the wanted head or tail entity.
    :param labels: Required when computing filtered ranks. One or zero label tensor of shape [B, E].
                   A 1 states that this entity is also a correct target for that triple. Is computed
                   across train, validation, and test set.
    :param filtered: Toggles filtered and raw rank computation. When computing filtered ranks, the
                     labels tensor must be passed.

    :return: A list with the ranks of the target entity in the score tensor. The list is of length [B]
    """

    if filtered:
        for index, target in enumerate(targets):
            # TODO: check that we dont change original labels, would be bad.
            labels[index, target] = 0  # we dont want to filter the right entity
            mask = labels[index] == 1
            scores[index][mask] = -np.inf  # assign worst possible score to filtered entities

    scores, entity_ranks = torch.sort(scores, descending=True)

    # target filtering shi
    # TODO

    # the final target ranks
    ranks = []

    for index, target in enumerate(targets):

        # TODO some target filtering code is missing

        # finally get the rank of the target
        target_rank = np.where(entity_ranks[index] == target)[0][0]
        ranks.append(target_rank + 1)  # ranks start at 1

    return ranks


def hits_at(k: int, ranks: List[float]) -> float:
    """

    :param k:
    :param ranks:
    :return:
    """
    if not ranks:
        return np.nan

    return len([r for r in ranks if r <= k]) / len(ranks) * 100


def mrr(ranks: List[float]) -> float:
    """
    Computes the mean reciprocal rank from a list of ranks.
    """
    if not ranks:
        return np.nan

    return np.mean(1 / np.asarray(ranks)) * 100


def mr(ranks: List[float]) -> int:
    """
    Computes the mean rank metric.
    """
    if not ranks:
        return np.nan

    return int(np.mean(ranks).round())


def evaluate_closed_world(model, preprocessed_data, batch_size, dataset_name="validation",
                          tail_prediction=True, head_prediction=True):

    logger.info("Starting evaluation on {} set.".format(dataset_name))

    tail_ranks_filtered, tail_ranks_raw = [], []
    head_ranks_filtered, head_ranks_raw = [], []

    total = preprocessed_data.num_triples // batch_size

    with torch.no_grad():
        for heads, relations, tails, head_labels, tail_labels in tqdm(preprocessed_data.iter_triples(batch_size),
                                                                      total=total, desc="Batches: "):

            if tail_prediction and tail_labels is not None:
                tail_scores = model(heads=heads, relations=relations).data.cpu()  # [B, E]
                tail_ranks_raw += compute_target_ranks(tail_scores, targets=tails, labels=None, filtered=False)
                tail_ranks_filtered += compute_target_ranks(tail_scores, targets=tails,
                                                            labels=tail_labels, filtered=True)

            if head_prediction and head_labels is not None:
                head_scores = model(heads=None, relations=relations, tails=tails).data.cpu()  # [B, E]
                head_ranks_raw += compute_target_ranks(head_scores, targets=heads, labels=None, filtered=False)
                head_ranks_filtered += compute_target_ranks(head_scores, targets=heads,
                                                            labels=head_labels, filtered=True)
            # from IPython import embed; embed()
    if tail_prediction:
        logger.info("Tail prediction:")
        logger.info(f"[Eval: {dataset_name}] {hits_at(1, tail_ranks_filtered):6.2f} Hits@1 (%)")
        logger.info(f"[Eval: {dataset_name}] {hits_at(3, tail_ranks_filtered):6.2f} Hits@3 (%)")
        logger.info(f"[Eval: {dataset_name}] {hits_at(10, tail_ranks_filtered):6.2f} Hits@10 (%)")
        logger.info(f"[Eval: {dataset_name}] {mrr(tail_ranks_filtered):6.2f} MRR (filtered) (%)")
        logger.info(f"[Eval: {dataset_name}] {mrr(tail_ranks_raw):6.2f} MRR (raw) (%)")
        logger.info(f"[Eval: {dataset_name}] Mean rank: {mr(tail_ranks_filtered)}")
        logger.info(f"[Eval: {dataset_name}] Mean rank raw: {mr(tail_ranks_raw)}")
    if head_prediction:
        logger.info("Head prediction:")
        logger.info(f"[Eval: {dataset_name}] {hits_at(1, head_ranks_filtered):6.2f} Hits@1 (%)")
        logger.info(f"[Eval: {dataset_name}] {hits_at(3, head_ranks_filtered):6.2f} Hits@3 (%)")
        logger.info(f"[Eval: {dataset_name}] {hits_at(10, head_ranks_filtered):6.2f} Hits@10 (%)")
        logger.info(f"[Eval: {dataset_name}] {mrr(head_ranks_filtered):6.2f} MRR (filtered) (%)")
        logger.info(f"[Eval: {dataset_name}] {mrr(head_ranks_raw):6.2f} MRR (raw) (%)")
        logger.info(f"[Eval: {dataset_name}] Mean rank: {mr(head_ranks_filtered)}")
        logger.info(f"[Eval: {dataset_name}] Mean rank raw: {mr(head_ranks_raw)}")
