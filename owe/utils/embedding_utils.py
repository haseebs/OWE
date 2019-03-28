# Note: Some functions in this file will have to be adapted to work with
# models other than complEx

import itertools
import logging
import random

from annoy import AnnoyIndex
import numpy as np
import torch

from owe.config import Config
from owe.utils import Statistics

logger = logging.getLogger("owe")


def dump_embeddings(epoch, target_dataset, target_emb, train_dataset,
                    writer, model, mapper, vocab, tag=''):
    """
    Generates the embeddings for tensorboard.

    :param epoch: current epoch
    :param target_dataset: dataset of which embeddings will be combined
                           with train embeddings
    :param target_emb: embeddings of entities from the target dataset
    :param train_dataset: dataset containing training entities
    :param writer: tensorboard writer
    :param model: OWE model
    :param mapper: OWE mapper
    :param vocab: vocabulary object
    :param tag: Name for the embedding
    :return:
    """
    if not Config.get("GetTensorboardEmbeddings"):
        return
    index2name, embeddings, embeddings_i = get_embeddings(model=model,
                                                          mapper=mapper,
                                                          vocab=vocab,
                                                          target_dataset=target_dataset,
                                                          target_emb=target_emb,
                                                          train_dataset=train_dataset,
                                                          get_string_labels=True,
                                                          do_transform=(epoch != 0))
    writer.add_embedding(torch.tensor(embeddings),
                         global_step=epoch,
                         tag=tag,
                         metadata_header=["Entity", "Source"],
                         metadata=[f.split('\t') for f in index2name.split(
                             '\n')[0:embeddings.shape[0]]])


def dump_tsv(epoch, dataset, test_emb, model, mapper, vocab):
    """
    Generates tab delimited embeddings and labels as accepted by tensorflow
    embedding projector (old).

    :param epoch: current epoch
    :param dataset: dataset object
    :param test_emb: embeddings of entities from the test dataset
    :param model: OWE model
    :param mapper: OWE mapper
    :param vocab: vocabulary object
    :return:
    """
    index2name, embeddings, embeddings_i = get_embeddings(model=model,
                                                          mapper=mapper,
                                                          vocab=vocab,
                                                          target_dataset=dataset.test,
                                                          target_emb=test_emb,
                                                          train_dataset=dataset.train,
                                                          get_string_labels=True,
                                                          do_transform=(epoch != 0))
    np.savetxt("embeddings_" + str(epoch) + ".tsv", embeddings, delimiter="\t")
    np.savetxt("embeddings_i_" + str(epoch) + ".tsv", embeddings_i, delimiter="\t")
    with open("labels_" + str(epoch) + ".tsv", "w") as f:
        f.write(index2name)


def get_embeddings(model, mapper, vocab, num_targets=None, target_dataset=None,
                   target_emb=None, train_dataset=None,
                   get_string_labels=False, do_transform=True,
                   combine_with_target=True, ):
    """
    Returns the required embeddings for the provided entities.
    :param model: OWE model
    :param mapper: OWE mapper
    :param vocab: vocabulary object
    :param num_targets: Number of entities from the target dataset to
                        include. None to include all
    :param target_dataset: Dataset of which embeddings will be combined
                           with train embeddings
    :param target_emb: embeddings of entities from the target dataset
    :param train_dataset: dataset containing training entities
    :param get_string_labels: True to return tab delimited string, False
                              to return a dict
    :param do_transform: Whether the target embeddings should be transformed
                         to complex space
    :param combine_with_target: Whether the target and train embeddings
                                should be combined or returned separately
    :return: index2name, embeddings_r and embeddings_i if combine_with_target is true
             otherwise return transformed_embeddings_r, transformed_embeddings_i,
             kgc_embeddings_r, kgc_embeddings_i
    """

    if Config.get("LinkPredictionModelType") != "ComplEx":
        raise NotImplementedError("This function does not work with TransE or something else yet")
    device = Config.get("device")

    train_subjects = list(set(train_dataset.heads.numpy()))
    target_subjects = list(set(target_dataset.heads.numpy()))

    if num_targets is not None:
        random.shuffle(target_subjects)
        target_subjects = target_subjects[0:num_targets - 1]
    else:
        num_targets = len(target_subjects)

    # Get entity info
    index2name = None
    if combine_with_target:
        if not get_string_labels:
            index2name = {}
            for i, k in enumerate(itertools.chain(target_subjects, train_subjects)):
                index2name[i] = "LABEL: " + str(vocab.index2entity[k].label_uscored_cased) \
                                + "\t DESCRIPTION: " + str(vocab.index2entity[k].description) \
                                + "\t SOURCE_SET: " + ("TARGET" if i <= num_targets else "TRAINING") \
                                + "\t TOKEN_FOUND: " + str(vocab.index2entity[k].exists_as_token)
        else:
            index2name = ""
            for i, k in enumerate(itertools.chain(target_subjects, train_subjects)):
                index2name += vocab.index2entity[k].label_uscored_cased
                index2name += "\tTARGET\n" if i < num_targets else "\tTRAINING\n"

    # Get transformed embeddings for target entities
    target_emb = torch.FloatTensor(np.asarray([target_emb[x] for x in target_subjects])).to(device)
    if do_transform:
        transformed_target_emb = mapper.module.transformer_r(target_emb)
        transformed_target_emb_i = mapper.module.transformer_i(target_emb)
    else:
        transformed_target_emb = target_emb
        transformed_target_emb_i = target_emb

    # Get complex embeddings for KB entities
    kb_emb_train = model.module.entity_embedding_r(torch.LongTensor(train_subjects).to(device))
    kb_emb_train_i = model.module.entity_embedding_i(torch.LongTensor(train_subjects).to(device))
    if combine_with_target:
        all_embeddings = torch.cat((transformed_target_emb, kb_emb_train), 0).data.cpu().numpy()
        all_embeddings_i = torch.cat((transformed_target_emb_i, kb_emb_train_i), 0).data.cpu().numpy()
        return index2name, all_embeddings, all_embeddings_i
    else:
        return transformed_target_emb.data.cpu().numpy(), \
               transformed_target_emb_i.data.cpu().numpy(), \
               kb_emb_train.data.cpu().numpy(), kb_emb_train_i.data.cpu().numpy()


def build_nn_index(embeddings, num_print=10, top_k=10, index2name=None,
                   name="neighbours.real", epoch=0, save=False, verbose=False):
    """
    Build and return the nearest neighour annoy index

    :param embeddings: Embeddings to add to the index
    :param num_print: Number of target entities whose neighbours should
                be printed
    :param top_k: Number of nearest neighbours to be printed for each entity
    :param index2name: Dict mapping annoy index values to entity
                       descriptions
    :param name: Name for the index file when saving
    :param epoch: Current epoch
    :param save: Whether the index should be saved to disk
    :param verbose: Whether to print the neighbours
    """
    index = AnnoyIndex(embeddings.shape[1])
    for i in range(len(embeddings)):
        index.add_item(i, embeddings[i])
    index.build(50)
    if save:
        index.save(name + str(epoch))
    if verbose:
        for k in range(num_print):
            logger.info("==================================================")
            logger.info(index2name[k])
            for j in index.get_nns_by_item(k, top_k):
                logger.info(index2name[j])
    return index


def find_neighbours(target_dataset, target_emb, train_dataset,
                    model, mapper, vocab, num_targets=None, top_k=10, num_print=10,
                    verbose=True, epoch=0):
    """
    Find nearest neighbours of random num_print transfromed target subject
    word embeddings and graph embeddings of training subjects
    :param target_dataset: dataset of which embeddings will be combined
                           with train embeddings
    :param target_emb: embeddings of entities from the target dataset
    :param train_dataset: dataset containing training entities
    :param model: OWE model
    :param mapper: OWE mapper
    :param vocab: vocabulary object
    :param num_targets: Number of entities from the target dataset to
                        include. None to include all
    :param top_k: Number of nearest neighbours to be printed for each entity
    :param num_print: Number of target entities whose neighbours should
                be printed
    :param verbose: Whether to print the neighbours
    :param epoch: Current epoch
    :return:
    """
    logger.info("Building an annoy index")

    index2name, all_embeddings, all_embeddings_i = get_embeddings(model=model,
                                                                  mapper=mapper,
                                                                  vocab=vocab,
                                                                  num_targets=num_targets,
                                                                  target_dataset=target_dataset,
                                                                  target_emb=target_emb,
                                                                  train_dataset=train_dataset,
                                                                  do_transform=(epoch != 0))
    build_nn_index(all_embeddings, index2name=index2name, epoch=epoch,
                   num_print=num_print, top_k=top_k, verbose=verbose)
    build_nn_index(all_embeddings_i, index2name=index2name, epoch=epoch,
                   name="neighbours.img", verbose=False)


def calculate_nn_ranks(epoch, train_dataset, train_emb, model, mapper, stat, vocab):
    """
    Calculates the nearest neighbour mean rank of complex train subject
    entity embeddings against the transformed train subject entity embeddings

    :param epoch: current epoch
    :param train_dataset: training dataset
    :param train_emb: embeddings for training set entities
    :param model: OWE model
    :param mapper: OWE mapper (use latest here if called at every epoch)
    :param stat: Statistics object to report stats
    :param vocab: vocabulary object
    :return:
    """

    def calculate_stats(trfm_emb, cplx_emb):
        ranks = []
        index = build_nn_index(trfm_emb)
        nn = [index.get_nns_by_vector(k, 1000) for k in cplx_emb]
        for i, k in enumerate(nn):
            rank = np.where(np.asarray(k) == i)[0]
            if rank.size == 0:
                rank = [len(trfm_emb)]
            ranks.append(1 + rank[0])
        mrr = np.divide(1, ranks)
        return rank, mrr

    logger.info("Calculating mean ranks for nearest neighbours")
    # DONT Use the best mapper, use latest here
    trfm_emb, trfm_emb_i, cplx_emb, cplx_emb_i = get_embeddings(model=model,
                                                                mapper=mapper,
                                                                vocab=vocab,
                                                                target_dataset=train_dataset,
                                                                target_emb=train_emb,
                                                                train_dataset=train_dataset,
                                                                combine_with_target=False)
    # Real neighbours
    ranks, mrr = calculate_stats(trfm_emb, cplx_emb)
    stat.record(nn_mean_rank=np.asarray(ranks).mean())
    stat.record(nn_mrr=mrr.mean())
    # Imaginary neighbours
    ranks_i, mrr_i = calculate_stats(trfm_emb_i, cplx_emb_i)
    stat.record(nn_mean_rank_i=np.asarray(ranks_i).mean())
    stat.record(nn_mrr_i=mrr_i.mean())
    stat.push_nn_ranks(epoch)
