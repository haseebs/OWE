import torch
import pickle
import logging

from owe.models import KGCBase

logger = logging.getLogger("owe")


class RotatE(KGCBase):

    def __init__(self, num_entities, num_relations, embedding_dim, gamma: float = 9.0):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embedding_r = torch.nn.Embedding(self.E, self.D)
        self.entity_embedding_i = torch.nn.Embedding(self.E, self.D)
        self.relation_embedding = torch.nn.Embedding(self.R, self.D)

        self.gamma = gamma
        self.epsilon = 2.0  # hardcoded
        self.embedding_range = (self.gamma + self.epsilon) / self.D

    def _score(self, heads_r, heads_i, relations_r, relations_i, tails_r, tails_i, tail_pred: bool):
        if tail_pred:
            re_score = heads_r * relations_r - heads_i * relations_i  # [B,1,D]
            im_score = heads_r * relations_i + heads_i * relations_r  # [B,1,D]
            re_score = re_score - tails_r
            im_score = im_score - tails_i
        else:
            re_score = relations_r * tails_r + relations_i * tails_i
            im_score = relations_r * tails_i - relations_i * tails_r
            re_score = re_score - heads_r
            im_score = im_score - heads_i

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma - score.sum(dim=2)
        return score

    def init_embeddings(self, dataset, emb_dir, entity2id="entity2id.txt",
                        relation2id="relation2id.txt"):
        """
        Initializes the rotate model with embeddings from previous runs.

        :param dataset:
        :param emb_dir:
        :param entity2id:
        :param relation2id:
        :return:
        """
        logger.info("Loading pretrained embeddings from %s into RotatE model" % str(emb_dir))
        entity_file = emb_dir / "entities.p"
        relation_file = emb_dir / "relations.p"

        entity2id_file = emb_dir / entity2id
        relation2id_file = emb_dir / relation2id

        if not (entity_file.exists()
                or relation_file.exists()
                or entity2id_file.exists() or relation2id_file.exists()):
            m = ("Trying to load pretrained embeddings (Config setting:"
                 "InitializeWithPretrainedComplexEmbedding ). Not all files"
                 "found under %s" % str(emb_dir))
            logger.error(m)
            raise ValueError(m)

        # mid -> int
        try:
            external_entity2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                                  for l in entity2id_file.open("rt")}
            external_relation2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                                    for l in relation2id_file.open("rt")}
        except IndexError:
            # First line contains header with amount of entities, first split fails
            external_entity2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                                  for l_no, l in enumerate(entity2id_file.open("rt")) if l_no >= 1}
            external_relation2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                                    for l_no, l in enumerate(relation2id_file.open("rt")) if l_no >= 1}

        our_entity2id = {e.entity_id: i for e, i in dataset.vocab.entity2id.items()}
        our_relation2id = dataset.vocab.relation2id


        try:
            emb = pickle.load(entity_file.open("rb"))
        except pickle.UnpicklingError:
            import numpy as np
            emb = np.load(entity_file)

        entity_r, entity_i = np.split(emb, 2, axis=1)
        self.copy_embeddings(self.entity_embedding_r, entity_r, external_entity2id, our_entity2id)
        self.copy_embeddings(self.entity_embedding_i, entity_i, external_entity2id, our_entity2id)
        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])
        try:
            emb = pickle.load(relation_file.open("rb"))
        except pickle.UnpicklingError:
            import numpy as np
            emb = np.load(relation_file)
        self.copy_embeddings(self.relation_embedding, emb, external_relation2id, our_relation2id)

    def score(self, heads, relations, tails):
        """
                :param relations: [B] Tensor of relations
                :param heads: [B], or [E] if predicting tails
                :param tails: [B], or [E] if predicting tails
                :return: [B] or [B,E] Tensor of predictions.
                """

        tail_pred = tails.size(0) == self.E
        assert (not tail_pred) == (heads.size(0) == self.E)

        if tail_pred:
            tails_r = self.entity_embedding_r(tails)  # [E] -> [E,D]
            tails_i = self.entity_embedding_i(tails)  # [E] -> [E,D]
            tails_r = tails_r.unsqueeze(0)  # [E,D] -> [1,E,D]
            tails_i = tails_i.unsqueeze(0)  # [E,D] -> [1,E,D]

            if self.embeddings is None:  # the head entity is known (part of the training graph)
                heads_r = self.entity_embedding_r(heads)  # [B] -> [B,D]
                heads_i = self.entity_embedding_i(heads)  # [B] -> [B,D]
            else:  # Use the projected head embeddings
                heads_r, heads_i = self.embeddings  # [B,D]
            heads_r = heads_r.unsqueeze(1)  # [B,D] -> [B,1,D]
            heads_i = heads_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        else:  # Head prediction
            heads_r = self.entity_embedding_r(heads)  # [E] -> [E,D]
            heads_i = self.entity_embedding_i(heads)  # [E] -> [E,D]
            heads_r = heads_r.unsqueeze(0)  # [E,D] -> [1,E,D]
            heads_i = heads_i.unsqueeze(0)  # [E,D] -> [1,E,D]

            if self.embeddings is None:  # We know the tail entity  # TODO remove self.embedding
                tails_r = self.entity_embedding_r(tails)  # [B] -> [B,D]
                tails_i = self.entity_embedding_i(tails)  # [B] -> [B,D]
            else:  # Use the projected tail embeddings
                tails_r, tails_i = self.embeddings  # [B,D]
            tails_r = tails_r.unsqueeze(1)  # [B,D] -> [B,1,D]
            tails_i = tails_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        relations = self.relation_embedding(relations)  # [B] -> [B,D]
        relations = relations.unsqueeze(1)  # [B,D] -> [B,1,D]

        pi = 3.14159262358979323846
        phase_relation = relations / (self.embedding_range / pi)
        relations_r = torch.cos(phase_relation)
        relations_i = torch.sin(phase_relation)

        # 1. Head Prediction: [1,E, D] * [B,1,D] * [B,1,D] -> [B, E]
        # 2. Tail Prediction: [B,1,D] * [B,1,D] * [1,E,D] -> [B, E]

        scores = self._score(heads_r, heads_i,
                             relations_r, relations_i,
                             tails_r, tails_i, tail_pred)

        return scores

