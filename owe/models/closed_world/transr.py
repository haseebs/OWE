import torch
import pickle
import logging

from owe.models import KGCBase

logger = logging.getLogger("owe")


class TransR(KGCBase):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransR, self).__init__(num_entities, num_relations, embedding_dim)
        self.entity_embedding = torch.nn.Embedding(self.E, self.D)

        self.DR = self.D  # Relation embedding dim, can be different from entity embedding for this model
        self.relation_embedding = torch.nn.Embedding(self.R, self.DR)
        self.transfer_embedding = torch.nn.Embedding(self.R, self.DR*self.D)

    def _transfer(self, transfer_matrix, embeddings):
        return torch.matmul(transfer_matrix, embeddings).squeeze()

    def _calc(self, h, r, t):
        return torch.abs(h + r - t)

    def init_embeddings(self, dataset, emb_dir, entity2id="entity2id.txt",
                        relation2id="relation2id.txt"):
        """
        Initializes the complex modelself, embeddings from previous runs.

        :param model:
        :param emb_dir:
        :param entity2id:
        :return:
        """
        logger.info("Loading pretrained embeddings from %s into TransR model" % str(emb_dir))

        import pickle
        entity_file = emb_dir / "entities.p"
        relation_file = emb_dir / "relations.p"
        transfer_file = emb_dir / "transfers.p"
        entity2id_file = emb_dir / entity2id
        relation2id_file = emb_dir / relation2id

        if not (entity_file.exists() or relation_file.exists() or transfer_file.exists()
                or entity2id_file.exists() or relation2id_file.exists()):
            m = ("Trying to load pretrained embeddings (Config setting:"
                 "InitializeWithPretrainedComplexEmbedding ). Not all files"
                 "found under %s" % str(emb_dir))
            logger.error(m)
            raise ValueError(m)

        # mid -> int
        external_entity2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                        for l in entity2id_file.open("rt")}
        our_entity2id = {e.key:i for e,i in dataset.vocab.entity2index.items()}
        external_relation2id = {l.split("\t")[0]: int(l.split("\t")[1].strip())
                            for l in relation2id_file.open("rt")}
        our_relation2id = dataset.vocab.relation2index

        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])
        emb = pickle.load(entity_file.open("rb"))
        self.copy_embeddings(self.entity_embedding, emb, external_entity2id, our_entity2id)
        # logger.info("First entity embedding row: %s" % model.model.module.entity_embedding.weight.data[0])

        # model.model.module.entity_embedding.weight.data.copy_(torch.from_numpy(emb))
        emb = pickle.load(relation_file.open("rb"))
        self.copy_embeddings(self.relation_embedding, emb, external_relation2id, our_relation2id)

        emb = pickle.load(transfer_file.open("rb"))
        self.copy_embeddings(self.transfer_embedding, emb, external_relation2id, our_relation2id)


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
            tail_prediction = False         # --> heads.shape = [E], tails.shape = [B]
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

        # from IPython import embed; embed()
        ####
        transfer_m = self.transfer_embedding(relations).view(-1, self.DR, self.D)  # [B, DR, D]
        relations = self.relation_embedding(relations)                          # [B, DR]

        heads = self._transfer(transfer_m, heads.unsqueeze(-1))  # [B, DR, D] * # [B, D, 1] -> [B, DR] (or E)
        tails = self._transfer(transfer_m.to("cpu").unsqueeze(1), tails.unsqueeze(-1).unsqueeze(0).to("cpu"))  # [B, 1, DR, D] * # [1, E, D, 1] -> [E, DR] (or E)
        tails = tails.to(self.device)  # [B, E, DR]
        #####

        if tail_prediction:
            heads = self._transfer(transfer_m, heads.unsqueeze(-1))             # [B, DR, D] * # [B, D, 1] -> [B, DR] (or E)
            tails = self._transfer(transfer_m.to("cpu").unsqueeze(1),
                                   tails.unsqueeze(-1).unsqueeze(0).to("cpu"))  # [B, 1, DR, D] * # [1, E, D, 1] -> [E, DR] (or E)
            tails = tails.to(self.device)       # [B, E, DR]

            heads = heads.unsqueeze(1)          # [B,DR] -> [B,1,DR]
            relations = relations.unsqueeze(1)  # [B,DR] -> [B,1,DR]
        else:
            tails = self._transfer(transfer_m, tails.unsqueeze(-1))  # [B, DR, D] * # [B, D, 1] -> [B, DR] (or E)
            heads = self._transfer(transfer_m.to("cpu").unsqueeze(1),
                                   heads.unsqueeze(-1).unsqueeze(0).to("cpu"))  # [B, 1, DR, D] * # [1, E, D, 1] -> [B, E, DR] (or E)
            heads = heads.to(self.device)  # [B, E, DR]

            tails = tails.unsqueeze(1)          # [B,DR] -> [B,1,DR]
            relations = relations.unsqueeze(1)  # [B,DR] -> [B,1,DR]

        scores = self._calc(heads, relations, tails).sum(dim=-1)

        return scores
