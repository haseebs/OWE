import copy
import logging
from pathlib import Path

from annoy import AnnoyIndex
import numpy as np
from tabulate import tabulate
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import linregress

from owe.config import Config
from owe.data import RelationalDataset
from owe.models import Mapper
from owe.models import KGCBase
from owe.utils import Statistics, save_checkpoint, decay_lr, sigmoid, cosine_loss
from owe.utils import dump_embeddings, find_neighbours, calculate_nn_ranks

logger = logging.getLogger("owe")


class OpenWorldPrediction:
    def __init__(self, kgc_model: KGCBase, dataset: RelationalDataset,
                 embedding_dim: int = None, outdir: Path = None, writer: SummaryWriter = None):
        self.best_epoch = 0
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.gpu = Config.get("GPUs")
        self.writer = writer
        self.dataset = dataset
        self.vocab = self.dataset.vocab
        self.embedding_dim = embedding_dim
        self.stat = Statistics(outdir, writer=writer)

        # Use the pretrained word embeddings to load embeddings of all
        # words present in the vocabulary
        # logger.info("Building embedding matrix")
        # self.vocab.get_emb_keys(text_embedding)

        self.kgc_model = kgc_model
        self.mapper = Mapper(self.vocab.vectors, self.embedding_dim)
        self.mapper_best = self.mapper
        self.optimizer = torch.optim.Adam(self.mapper.parameters(),
                                          lr=Config.get("LearningRate"))

        if not Config.get("DisableCuda"):
            self.kgc_model = torch.nn.DataParallel(self.kgc_model, device_ids=self.gpu)
            self.kgc_model = self.kgc_model.to(Config.get("device"))
            self.mapper = torch.nn.DataParallel(self.mapper, device_ids=self.gpu)
            self.mapper = self.mapper.to(Config.get("device"))

        # Get the embeddings of entities that are present in the respective datasets
        # self.train_emb = dataset.train.get_emb_dict(text_embedding,
        #                                             Config.get("UseTailsToOptimize"))
        # self.val_emb = dataset.validation.get_emb_dict(text_embedding, False)
        # self.test_emb = dataset.test.get_emb_dict(text_embedding, False)

        # Load the KGC embeddings from the ones trained by OpenKE
        # if Config.get("InitializeWithPretrainedKGCEmbedding"):
        #     self.kgc_model.module.init_embeddings(self.dataset,
        #                                           self.outdir / "embeddings",
        #                                       "entity2id.txt")

    def stop_early(self, epoch):
        def check_threshold(metric, threshold=0.001, last_x=5):
            """
            We stop if the slope of the line that fits the last_x values gets
            below the threshold.

            :param metric:
            :param threshold:
            :param last_x:
            :return:
            """
            y = metric[-last_x:]
            x = np.arange(len(metric))[-last_x:]
            slope, _, _, _, _ = linregress(x, y)
            return True if slope < threshold else False

        if Config.get("EarlyStopping") and \
                epoch > Config.get("EarlyStoppingMinEpochs") and \
                check_threshold(self.stat.mean_rec_rank_filtered,
                                threshold=Config.get("EarlyStoppingThreshold"),
                                last_x=Config.get("EarlyStoppingLastX")):
            logger.info("[EarlyStopping] Improvement of MRR over the last %s "
                        "epochs less than %s. Stopping training!" % (
                            Config.get("EarlyStoppingLastX"),
                            Config.get("EarlyStoppingThreshold")))
            self.mapper_best = copy.deepcopy(self.mapper)
            self.best_epoch = epoch
            return True
        else:
            return False

    def evaluate(self, preprocessed_data, dataset_name="validation",
                 predict_tails=True, predict_heads=False, print_predictions=False,
                 epoch=0, print_nn_in_train=False):

        def make_tabular(h, r, t, sorted_pred, ranked_entity, correct_rank, top_k=10):
            """
            :param sorted_pred: [E]
            :param ranked_entity:
            :return:
            """
            if Config.get("UseTargetFilteringShi"):
                ranked_entity = np.ma.masked_inside(ranked_entity, 2, -1)
                sorted_pred = np.ma.masked_array(sorted_pred,
                                                 mask=ranked_entity.mask,
                                                 fill_value=-9999999).compressed()
                ranked_entity = ranked_entity.compressed()
            return (str(self.vocab.id2entity[h.item()]),
                    self.vocab.id2relation[r.item()],
                    "{} ({})".format(str(self.vocab.id2entity[t.item()]), correct_rank),
                    ["({}, {:.2f})".format(str(self.vocab.id2entity[e_i.item()]),
                                           sigmoid(p.item()))
                     for p, e_i in zip(sorted_pred[:top_k], ranked_entity[:top_k])])

        def filter_targets_shi(ranked_tail_entity):
            mask = np.ndarray(ranked_tail_entity.shape)
            for head, tail_preds in enumerate(ranked_tail_entity):
                mask[head] = ~np.in1d(tail_preds.numpy(),
                                      list(self.dataset.train.rel2tails[relations[head].item()]))
            return np.ma.masked_array(ranked_tail_entity, mask=mask, fill_value=-1)

        if print_nn_in_train:
            logger.info("Building Annoy Index")
            index_r = AnnoyIndex(self.embedding_dim)
            index_i = AnnoyIndex(self.embedding_dim)

            for i, vec in enumerate(self.kgc_model.module.entity_embedding_r.weight):
                index_r.add_item(i, vec)
            for i, vec in enumerate(self.kgc_model.module.entity_embedding_i.weight):
                index_i.add_item(i, vec)

            index_r.build(2000)
            index_i.build(2000)
            logger.info("Done")

        self.mapper.eval()
        if dataset_name == "validation":
            if Config.get("CalculateNNMeanRank"):
                calculate_nn_ranks(epoch, self.dataset.train,
                                   self.train_emb, self.kgc_model,
                                   self.mapper, self.stat, self.vocab)
            if Config.get("PrintTrainNN"):
                logger.info("Nearest neighbours where TARGET=TRAIN")
                find_neighbours(target_dataset=self.dataset.train,
                                target_emb=self.train_emb,
                                train_dataset=self.dataset.train,
                                model=self.kgc_model,
                                mapper=self.mapper,
                                vocab=self.vocab,
                                num_targets=10,
                                epoch=epoch)
            if Config.get("PrintTestNN"):
                logger.info("Nearest neighbours where TARGET=TEST")
                find_neighbours(target_dataset=self.dataset.test,
                                target_emb=self.train_emb,
                                train_dataset=self.dataset.train,
                                model=self.kgc_model,
                                mapper=self.mapper,
                                vocab=self.vocab,
                                num_targets=len(self.test_emb.keys()),
                                epoch=epoch)

        logger.info("Starting evaluation on {} set.".format(dataset_name))
        if epoch == 0:
            logger.info("Performing evaluation without using the transformation")
        tail_ranks = []
        tail_ranks_raw = []
        tabular_data = []  # (h,r,t,preds[:10])
        batch_size = 80
        total_iter = preprocessed_data.num_triples // batch_size
        iterator = zip(preprocessed_data.iter_triples(batch_size),
                       preprocessed_data.iter_entitydata_triplewise(batch_size,
                                                                    yield_heads=True,
                                                                    yield_tails=False))

        for (heads, relations, tails, _, tail_labels),  (heads2, head_names, head_desc) in tqdm(iterator,
                                                                                        total=total_iter,
                                                                                        desc="Evaluation: "):

            assert torch.equal(heads, heads2)  # check if iterators yield same data, otherwise its broken

            if predict_tails:

                if Config.get("LinkPredictionModelType") in ("ComplEx", "RotatE"):
                    transformed_embeddings_r, transformed_embeddings_i = self.mapper(head_names, head_desc)
                    embeddings = (transformed_embeddings_r, transformed_embeddings_i)
                elif Config.get("LinkPredictionModelType") in ("TransE", "TransR", "DistMult"):
                    transformed_embeddings = self.mapper(head_names, head_desc)
                    embeddings = transformed_embeddings
                else:
                    raise ValueError("LinkPredictionModelType unknown")

                if Config.get("EvalRandomHeads"):
                    random_batch_indices = torch.randperm(len(self.dataset.train.heads_unique))
                    heads = self.dataset.train.heads[random_batch_indices[0:len(heads)]]
                    tail_predictions = self.kgc_model(heads, relations).data.cpu()
                else:
                    tail_predictions = self.kgc_model(heads, relations,
                                                      embeddings=embeddings,
                                                      ).data.cpu()  # [B, E]
                    if Config.get("ShiTargetFilteringBaseline"):
                        # tail_predictions = tail_predictions[:, torch.randperm(tail_predictions.shape[1])]
                        tail_predictions = torch.rand(tail_predictions.shape)
                # evaluate raw
                raw_sorted_tail_preds, raw_ranked_tail_entity = torch.sort(
                    tail_predictions, descending=True)
                # evaluate tails filtered
                for index, tail in enumerate(tails):
                    tail_labels[index, tail] = 0  # we dont want to filter the right one
                    tail_predictions[index][tail_labels[index] == 1] = -np.inf
                filtered_sorted_tail_preds, filtered_ranked_tail_entity = torch.sort(
                    tail_predictions, descending=True)
                if Config.get("UseTargetFilteringShi"):
                    raw_ranked_tail_entity = filter_targets_shi(raw_ranked_tail_entity)
                    filtered_ranked_tail_entity = filter_targets_shi(filtered_ranked_tail_entity)
                # compute ranks
                for index, tail in enumerate(tails):
                    if Config.get("UseTargetFilteringShi"):
                        rank_e2_raw = np.where(raw_ranked_tail_entity[index].compressed() == tail.item())[0]
                        if rank_e2_raw.shape[0] == 0:
                            continue
                            rank_e2_raw = self.vocab.num_entities
                            rank_e2 = self.vocab.num_entities
                        else:
                            rank_e2_raw = rank_e2_raw[0]
                            rank_e2 = np.where(filtered_ranked_tail_entity[index].compressed() == tail.item())[0][0]
                    else:
                        rank_e2_raw = np.where(raw_ranked_tail_entity[index] == tail)[0][0]
                        rank_e2 = np.where(filtered_ranked_tail_entity[index] == tail)[0][0]
                    rank_e2_raw += 1
                    rank_e2 += 1
                    tail_ranks.append(rank_e2)
                    tail_ranks_raw.append(rank_e2_raw)
                    if rank_e2 > 0 and print_predictions:
                        tabular_data.append(make_tabular(heads[index],
                                                         relations[index],
                                                         tail,
                                                         filtered_sorted_tail_preds[index],
                                                         filtered_ranked_tail_entity[index],
                                                         rank_e2))

                        if print_nn_in_train:
                            neighbors_r = index_r.get_nns_by_vector(transformed_embeddings_r[index], 10)
                            neighbors_i = index_i.get_nns_by_vector(transformed_embeddings_i[index], 10)

                            tabular_data.append(("\n".join([str(self.vocab.index2entity[n]) for n in neighbors_r]),
                                                 "\n".join([str(self.vocab.index2entity[n]) for n in neighbors_i]),
                                                 "", ""), )
                            tabular_data.append((" ", " ", " ", " ",))

                # print 100 results at a time
                if print_predictions:
                    for chuck_s in range(0, len(tabular_data), 100):
                        d = tabular_data[chuck_s:(chuck_s + 100)]
                        logger.info(tabulate(d, headers=['Head', 'Relation', 'Tail (Rank)',
                                                         'Predictions']))
                    tabular_data = []
                # from IPython import embed; embed()

        self.stat.record(hitsat1=len([r for r in tail_ranks if r <= 1]) / len(tail_ranks))
        self.stat.record(hitsat3=len([r for r in tail_ranks if r <= 3]) / len(tail_ranks))
        self.stat.record(hitsat10=len([r for r in tail_ranks if r <= 10]) / len(tail_ranks))
        self.stat.record(mean_rank=int(np.mean(tail_ranks).round()))
        self.stat.record(mean_rank_raw=int(np.mean(tail_ranks_raw).round()))
        self.stat.record(mean_rec_rank_filtered=np.mean(1 / np.asarray(tail_ranks)))
        self.stat.record(mean_rec_rank_raw=np.mean(1 / np.asarray(tail_ranks_raw)))
        self.stat.push_eval(epoch, dataset_name, verbose=True)

    def train(self, epochs=1, batch_size=None, validate_every=10, start_epoch: int = 1):
        """
        Train the matrices for the transformation of word embeddings to kgc embedding

        :param epochs: total epochs to complete
        :param batch_size:
        :param validate_every: after how many epochs to evaluate on val split
        """

        self.evaluate(self.dataset.validation, "validation", True, False, epoch=0)

        if Config.get("Loss") == "Pairwise":
            criterion = nn.PairwiseDistance(p=2)
        elif Config.get("Loss") == "Cosine":
            criterion = cosine_loss
        else:
            raise ValueError("Wrong loss function config setting.")

        # Whether we should iterate triplets or entities
        # Iterating triplets produced better results in our experiments
        if Config.get("IterTriplets"):
            logging.info(f"Training for {epochs} by iterating {self.dataset.train.num_triples} triples with "
                         f"a batch size of {batch_size}.")
            iterator = self.dataset.train.iter_entitydata_triplewise
            num_batches = self.dataset.train.num_triples // batch_size
        else:
            logging.info(f"Training for {epochs} by iterating {self.dataset.train.num_entities} entities with "
                         f"a batch size of {batch_size}.")
            iterator = self.dataset.train.iter_entitydata_entitywise
            num_batches = self.dataset.train.num_entities // batch_size

        for epoch in range(start_epoch, epochs + 1):
            train_batch_losses = []
            decay_lr(current_epoch=epoch, optimizer=self.optimizer)
            for entities, names, descriptions in tqdm(iterator(batch_size,
                                                             yield_heads=True,
                                                             yield_tails=Config.get("UseTailsToOptimize")),
                                                    total=num_batches,
                                                    desc="Train: "):

                self.mapper.train()
                self.optimizer.zero_grad()
                entities = entities.to(Config.get("device"))

                # Compare transformed text-based emb with original kgc embedding
                # Different setting in case of ComplEx model due to two entity
                # embedding matrices.
                if Config.get("LinkPredictionModelType") in ("ComplEx", "RotatE"):
                    transformed_emb_r, transformed_emb_i = self.mapper(names, descriptions)
                    emb_r = self.kgc_model.module.entity_embedding_r(entities)
                    emb_i = self.kgc_model.module.entity_embedding_i(entities)
                    loss = criterion(emb_r, transformed_emb_r) + criterion(emb_i, transformed_emb_i)
                elif Config.get("LinkPredictionModelType") in ("TransE", "TransR", "DistMult"):
                    transformed_emb = self.mapper(names, descriptions)
                    model_emb = self.kgc_model.module.entity_embedding(entities)
                    loss = criterion(model_emb, transformed_emb)
                else:
                    raise ValueError("LinkPredictionModelType unknown")

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                train_batch_losses.append(loss.item())

            self.dataset.train.shuffle()
            self.stat.record(train_losses=np.mean(train_batch_losses))
            self.stat.push_losses(epoch, verbose=True)

            # Evaluate on validation set and save the checkpoint
            if (epoch - 1) % validate_every == 0:
                self.stat.record(metric_epochs=epoch)
                self.evaluate(self.dataset.validation,
                              "validation",
                              predict_tails=True,
                              predict_heads=False,
                              print_predictions=False,
                              epoch=epoch)
                save_checkpoint(epoch=epoch,
                                outdir=self.outdir,
                                model=self.kgc_model,
                                mapper=self.mapper,
                                optimizer=self.optimizer,
                                criterion=self.stat.mean_rec_rank_filtered)
                if self.stop_early(epoch):
                    break

        # Save embeddings for tensorboard embedding projector
        if Config.get("GetTensorboardEmbeddings"):
            dump_embeddings(epoch=self.best_epoch,
                            target_dataset=self.dataset.test,
                            target_emb=self.test_emb,
                            train_dataset=self.dataset.train,
                            writer=self.writer,
                            model=self.kgc_model,
                            mapper=self.mapper_best,
                            vocab=self.vocab,
                            tag="TARGET_IS_TEST")
            dump_embeddings(epoch=self.best_epoch,
                            target_dataset=self.dataset.train,
                            target_emb=self.test_emb,
                            train_dataset=self.dataset.train,
                            writer=self.writer,
                            model=self.kgc_model,
                            mapper=self.mapper_best,
                            vocab=self.vocab,
                            tag="TARGET_IS_TRAIN")
