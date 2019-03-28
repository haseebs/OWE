import torch
import pickle
import logging

from owe.models import KGCBase

logger = logging.getLogger("owe")


class TransE(KGCBase):
    """
    A simple transE model without training functionality
    """

    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embedding = torch.nn.Embedding(self.E, self.D)
        self.relation_embedding = torch.nn.Embedding(self.R, self.D)

    @staticmethod
    def _score(h, r, t):
        return torch.abs(h + r - t).sum(dim=-1).neg()

    def init_embeddings(self, dataset, emb_dir, entity2id="entity2id.txt",
                        relation2id="relation2id.txt"):
        """
        Initializes the complex model with embeddings from previous runs.

        :param dataset:
        :param emb_dir:
        :param entity2id:
        :param relation2id:
        :return:
        """
        logger.info("Loading pretrained embeddings from %s into TransE model" % str(emb_dir))

        entity_file = emb_dir / "entities.p"
        relation_file = emb_dir / "relations.p"
        entity2id_file = emb_dir / entity2id
        relation2id_file = emb_dir / relation2id

        if not (entity_file.exists() or relation_file.exists()
                or entity2id_file.exists() or relation2id_file.exists()):
            m = ("Trying to load pretrained embeddings (Config setting:"
                 "InitializeWithPretrainedComplexEmbedding ). Not all files"
                 "found under %s" % str(emb_dir))
            logger.error(m)
            raise ValueError(m)

        # mid -> int
        external_entity2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                              for l in entity2id_file.open("rt")}
        our_entity2id = {e.key: i for e, i in dataset.vocab.entity2index.items()}
        external_relation2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                                for l in relation2id_file.open("rt")}
        our_relation2id = dataset.vocab.relation2index

        emb = pickle.load(entity_file.open("rb"))
        self.copy_embeddings(self.entity_embedding, emb, external_entity2id, our_entity2id)
        emb = pickle.load(relation_file.open("rb"))
        self.copy_embeddings(self.relation_embedding, emb, external_relation2id, our_relation2id)

    def score(self, heads, relations, tails, predict=False):
        """
        Either heads or tails is of dim |E| when predicting otherwise batchsize |B|

        :param heads: [B] or [E] tensor of heads, if [E] then head prediction
        :param relations: [B] tensor of relations
        :param tails: [B] or [E] tensor of heads, if [E] then tail prediction
        :param predict: Must be true here, no training
        :return:
        """

        if not predict:
            raise NotImplementedError("This model can only do prediction")

        if relations.shape == heads.shape:  # --> tails.shape = [E], heads.shape = [B]
            tail_prediction = True
        elif relations.shape == tails.shape:
            tail_prediction = False  # --> heads.shape = [E], tails.shape = [B]
        else:
            raise ValueError("Size mismatch")

        # embed everything
        if self.embeddings is None:
            heads = self.entity_embedding(heads)
            tails = self.entity_embedding(tails)
        elif self.embeddings is not None and tail_prediction:
            heads = self.embeddings
            tails = self.entity_embedding(tails)
        elif self.embeddings is not None and not tail_prediction:
            heads = self.entity_embedding(heads)
            tails = self.embeddings

        relations = self.relation_embedding(relations)

        if tail_prediction:
            heads = heads.unsqueeze(1)  # [B,D] -> [B,1,D]
            relations = relations.unsqueeze(1)  # [B,D] -> [B,1,D]
            tails = tails.unsqueeze(0)  # [E,D] -> [1,E,D]
        else:
            heads = heads.unsqueeze(0)  # [E,D] -> [1,E,D]
            relations = relations.unsqueeze(1)  # [B,D] -> [B,1,D]
            tails = tails.unsqueeze(1)  # [B,D] -> [B,1,D]
        scores = self._score(heads, relations, tails)

        return scores
