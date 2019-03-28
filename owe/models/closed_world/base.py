import logging
import torch
from owe.config import Config

logger = logging.getLogger('owe')


class KGCBase(torch.nn.Module):

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()

        self.E = num_entities
        self.R = num_relations
        self.D = embedding_dim
        self.device = Config.get("device")
        self.embeddings = None  # Will be set in forward pass

        logger.info("Initialized %s model with %s entities, %s relations embedded with "
                    "%s dimensions" % (self.__class__.__name__, self.E, self.R, self.D))

    def copy_embeddings(self, our_emb, ex_embedding, external_e2id, our_e2id):
        """
        Used to copy trained KGC embeddings from OpenKE to our model.

        :param our_emb:
        :param ex_embedding:
        :param ex_embedding:
        :param external_e2id:
        :param our_e2id:
        """
        # TODO fill this docstring above
        logger.info("Loading from emb (%s) to our emb (%s)" % (ex_embedding.shape,
                                                               our_emb.weight.shape))
        # logger.info(len(our_e2id))
        c, s = 0, 0
        for e, e_id in external_e2id.items():
            if e not in our_e2id:
                logger.debug("skipped", e)
                s += 1
                continue
            i = our_e2id[e]
            # logger.info("id->id: %s->%s" % (e_id, i))
            try:
                our_emb.weight.data[i, :].copy_(torch.from_numpy(ex_embedding[e_id, :]))
            except IndexError:
                # not a train entity, only train entities fit in our embedding
                logger.debug("skipped", e)
                s += 1
                pass
            c += 1
        logger.info("Loaded %s/%s rows." % (c, c + s))

    def forward(self, heads: torch.Tensor = None, relations: torch.Tensor = None,
                tails: torch.Tensor = None, embeddings: torch.Tensor = None):
        """
        While prediction:
        :param heads: [1] Variable of entity ids
        :param relations: [1] Variable of relation ids
        :param tails: [E] Variable of tail ids
        :param embeddings: TODO

        Always:
        :return: [B] Tensor of scores
        """
        self.embeddings = embeddings

        if tails is None:
            tails = torch.arange(self.E).long()
        if heads is None:
            heads = torch.arange(self.E).long()

        heads, relations, tails = heads.to(self.device), relations.to(self.device), tails.to(self.device)
        score = self.score(heads, relations, tails)

        return score

    def score(self, heads, relations, tails):
        """
        Should be implemented by subclasses.

        :param heads: [B] Variable of entity ids
        :param relations: [B] Variable of relation ids
        :param tails: [B] Variable of tail ids

        :return: [B] Variable of scores
        """
        raise NotImplementedError
