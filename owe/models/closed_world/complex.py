import torch
import pickle
import logging

from owe.models import KGCBase

logger = logging.getLogger("owe")


class ComplEx(KGCBase):

    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embedding_r = torch.nn.Embedding(self.E, self.D)
        self.entity_embedding_i = torch.nn.Embedding(self.E, self.D)
        self.relation_embedding_r = torch.nn.Embedding(self.R, self.D)
        self.relation_embedding_i = torch.nn.Embedding(self.R, self.D)

    @staticmethod
    def _score(heads_r, heads_i, relations_r, relations_i, tails_r, tails_i):
        return ((heads_r * relations_r * tails_r) +
                (heads_i * relations_r * tails_i) +
                (heads_r * relations_i * tails_i) -
                (heads_i * relations_i * tails_r)).sum(dim=-1)

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
        logger.info("Loading pretrained embeddings from %s into ComplEx model" % str(emb_dir))

        entity_r_file = emb_dir / "entities_r.p"
        entity_i_file = emb_dir / "entities_i.p"
        relation_r_file = emb_dir / "relations_r.p"
        relation_i_file = emb_dir / "relations_i.p"
        entity2id_file = emb_dir / entity2id
        relation2id_file = emb_dir / relation2id

        if not (entity_r_file.exists() or entity_i_file.exists()
                or relation_r_file.exists() or relation_i_file.exists()
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

        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])
        emb = pickle.load(entity_r_file.open("rb"))
        self.copy_embeddings(self.entity_embedding_r, emb, external_entity2id, our_entity2id)
        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])

        # model.model.module.entity_embedding.weight.data.copy_(torch.from_numpy(emb))
        emb = pickle.load(entity_i_file.open("rb"))
        self.copy_embeddings(self.entity_embedding_i, emb, external_entity2id, our_entity2id)

        emb = pickle.load(relation_r_file.open("rb"))
        # model.model.module.weight.data.copy_(torch.from_numpy(emb))
        self.copy_embeddings(self.relation_embedding_r, emb, external_relation2id, our_relation2id)

        emb = pickle.load(relation_i_file.open("rb"))
        self.copy_embeddings(self.relation_embedding_i, emb, external_relation2id, our_relation2id)
        # model.model.module.relation_embedding_i.weight.data.copy_(torch.from_numpy(emb))

        # logger.info("Loaded embeddings from %s to complex model." % (emb_dir))

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

            if self.embeddings is None:  # the head entity is known (part of the training graph) # TODO remove self.embedding
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

        relations_r = self.relation_embedding_r(relations)  # [B] -> [B,D]
        relations_i = self.relation_embedding_i(relations)  # [B] -> [B,D]
        relations_r = relations_r.unsqueeze(1)  # [B,D] -> [B,1,D]
        relations_i = relations_i.unsqueeze(1)  # [B,D] -> [B,1,D]

        # 1. Head Prediction: [1,E, D] * [B,1,D] * [B,1,D] -> [B, E]
        # 2. Tail Prediction: [B,1,D] * [B,1,D] * [1,E,D] -> [B, E]
        scores = self._score(heads_r, heads_i,
                             relations_r, relations_i,
                             tails_r, tails_i)

        return scores
