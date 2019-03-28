import json
import logging
import re
import attr
from typing import List, Tuple, Optional, Dict, Union, Set, DefaultDict, Iterator, Iterable, DefaultDict

import torch
import numpy as np
import pathlib as pl
import multiprocessing as mp
from collections import defaultdict
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize


from owe.config import Config


logger = logging.getLogger("owe")


def tokenize(content: str, lower: bool = True, remove_punctuation: bool = True, max_len: int = 100000) -> List[str]:
    """
    Uses nltks word_tokenize with option to remove punctuation and limit the len,

    :param content: The string that shall be tokenized.
    :param lower: Lowers content string
    :param remove_punctuation: Removes single punctuation tokens and tokens that starts with a
    :param max_len: Maximum amount of tokens in result
    :return:
    """
    if not content:
        return []

    if not isinstance(content, str):
        raise ValueError("Content must be a string.")

    if lower:
        content = content.lower()

    if remove_punctuation:
        content = re.sub('[^A-Za-z0-9]+', '', content)

    content = word_tokenize(content)
    content = content[:max_len]
    return content


def load_triple_file(data_file: str, split_symbol: str = '\t', skip_header: bool = False) -> List[Tuple[str, str, str]]:
    logger.debug("Loading triples and relations from: %s" % data_file)

    triples = []
    with open(data_file, 'rt') as f:
        for line_nr, line in enumerate(f):
            if skip_header and line_nr == 0:
                continue

            h, r, t = line.strip().split(split_symbol)
            triples.append((h, r, t))

    logger.debug(f"Loaded {len(triples)} triples from {data_file}")
    return triples


def load_embedding_file(embedding_file: str) -> KeyedVectors:
    logger.info(f"Loading word vectors from: {embedding_file}...")
    return KeyedVectors.load_word2vec_format(embedding_file, binary=not embedding_file.endswith(".txt"))


class AugmentationStatistics:
    num_matched_ids = 0
    num_matched_wiki2vec_ids = 0
    num_matched_phrases = {level: 0 for level in range(6)}
    num_matched_wiki2vec_entities = {level: 0 for level in range(6)}  # augmentations in augement_token()
    num_unmatched = 0

    @classmethod
    def log(cls):
        logger.info(f"Matched entities with 'ID': {cls.num_matched_ids}")
        logger.info(f"Matched entities with 'ENTITIY/ID' (wiki2vec notation): {cls.num_matched_wiki2vec_ids}")
        logger.info(f"Matched entities with augmented phrase as "
                    f"wiki2vec entity (level:count): {cls.num_matched_wiki2vec_entities}")
        logger.info(f"Matched entities with augmented phrase (level:count): {cls.num_matched_phrases}")
        logger.info(f"Unmatched entities: {cls.num_unmatched}")

def find_phrase(label: str, embedding: Union[Set[str], KeyedVectors], entity_id: str = '') -> Optional[str]:
    def augment(token: str) -> Tuple[str, str, str, str, str, str]:
        """
        Augments a phrase (on spaces between tokens) and yields augmented strings
        in descending priority. Best match should be the first in tuple.
        """

        def _augment_token(token: str, remove_punctuation: bool,
                           lower: bool, capitalize_words: bool) -> str:
            """
            Adds underscores on whitespaces and augments the token.
            """
            if not token:
                return token

            if not isinstance(token, str):
                raise ValueError("Content must be a string.")

            if remove_punctuation:
                token = re.sub('[^A-Za-z0-9 ]+', '', token)

            if lower:
                token = token.lower()

            if capitalize_words:
                token = " ".join(w.capitalize() for w in token.split(' '))

            # Finally replace whitespaces with underscores
            token = re.sub(' ', '_', token)

            return token

        return (_augment_token(token, lower=False, remove_punctuation=False, capitalize_words=False),
                _augment_token(token, lower=True, remove_punctuation=False, capitalize_words=False),
                _augment_token(token, lower=False, remove_punctuation=False, capitalize_words=True),
                _augment_token(token, lower=False, remove_punctuation=True, capitalize_words=False),
                _augment_token(token, lower=True, remove_punctuation=True, capitalize_words=False),
                _augment_token(token, lower=False, remove_punctuation=True, capitalize_words=False))


    # Do various augmentations to match label or id in given embedding.
    # Best match would be `entity_id` (WARNING if integer IDs are used on
    # a dataset this could lead to wrong matches eg.: 1958)
    if entity_id and entity_id in embedding:
        AugmentationStatistics.num_matched_ids += 1
        return entity_id

    # Now lets first try to match wiki2vec's internal wikipedia article page
    # entities
    w2v_entity_prefix = 'ENTITY/'
    if entity_id and w2v_entity_prefix + entity_id in embedding:
        AugmentationStatistics.num_matched_wiki2vec_ids += 1
        return w2v_entity_prefix + entity_id

    augmentations = augment(label)
    for level, augmented_label in enumerate(augmentations):
        if w2v_entity_prefix + augmented_label in embedding:
            AugmentationStatistics.num_matched_wiki2vec_entities[level] += 1
            return w2v_entity_prefix + augmented_label  # Just return first match, bests are in first

    # Then match label without wiki2vec notation
    for level, augmented_label in enumerate(augmentations):
        if augmented_label in embedding:
            AugmentationStatistics.num_matched_phrases[level] += 1
            return augmented_label  # Just return first match, bests are in first

    AugmentationStatistics.num_unmatched += 1
    # If we couldn't find a phrase until we return None
    return


@attr.s(cmp=False)
class Entity:
    entity_id: str = attr.ib()  # /m/x1121x1
    name: str = attr.ib()  # Ursula Le Guin
    description: str = attr.ib(repr=False)  # Is a fantasy writer ...

    # Will be set later with embedding
    name_tokens: List[str] = attr.ib(init=False, repr=False)
    description_tokens: List[str] = attr.ib(init=False, repr=False)

    name_tokens_idx: torch.tensor = attr.ib(init=False, repr=False)
    description_tokens_idx: torch.tensor = attr.ib(init=False, repr=False)

    def init_tokens(self, embedding: KeyedVectors, match_phrase_token: bool = True) -> None:
        self.name_tokens, self.description_tokens = self.get_tokens_in_emb(embedding)

    def get_tokens_in_emb(self, embedding):
        def tokenize_old(content, lower=True, remove_punctuation=True, add_underscores=False, limit_len=100000):
            """
            Splits on spaces between tokens.

            :param content: The string that shall be tokenized.
            :param lower: Lowers content string
            :param remove_punctuation: Removes single punctuation tokens
            :param add_underscores: Replaces spaces with underscores
            :return:
            """
            if not content or not limit_len:
                return [""] if add_underscores else []

            if not isinstance(content, (str)):
                raise ValueError("Content must be a string.")

            if remove_punctuation:
                content = re.sub('[^A-Za-z0-9 ]+', '', content)

            if lower:
                content = content.lower()

            if add_underscores:
                res = [re.sub(' ', '_', content)]
                return res

            res = word_tokenize(content)

            res = res[:limit_len]
            return res

        label = tokenize_old(self.name)  # list
        label_uscored_uncased_punct = tokenize_old(self.name, remove_punctuation=False, add_underscores=True)[0]  # str
        label_uscored_cased_punct = tokenize_old(self.name, remove_punctuation=False, lower=False, add_underscores=True)[
            0]  # str
        label_uscored_uncased = tokenize_old(self.name, remove_punctuation=False, add_underscores=True)[0]  # str
        label_uscored_cased = tokenize_old(self.name, remove_punctuation=False, lower=False, add_underscores=True)[
            0]  # str

        description = tokenize_old(self.description, limit_len=Config.get("LimitDescription"))  # list
        # DEPRECATED should be removed
        if "ENTITY/" + self.entity_id in embedding:
            name_token = ["ENTITY/" + self.entity_id]
        elif self.entity_id in embedding:
            name_token = [self.entity_id]
        elif "ENTITY/" + label_uscored_cased_punct in embedding:
            name_token = ["ENTITY/" + label_uscored_cased_punct]
        elif "ENTITY/" + label_uscored_uncased_punct in embedding:
            name_token = ["ENTITY/" + label_uscored_uncased_punct]
        elif "ENTITY/" + label_uscored_cased in embedding:
            name_token = ["ENTITY/" + label_uscored_cased]
        elif "ENTITY/" + label_uscored_uncased in embedding:
            name_token = ["ENTITY/" + label_uscored_uncased]
        elif label_uscored_cased_punct in embedding:
            name_token = [label_uscored_cased_punct]
        elif label_uscored_uncased_punct in embedding:
            name_token = [label_uscored_uncased_punct]
        elif label_uscored_cased in embedding:
            name_token = [label_uscored_cased]
        elif label_uscored_uncased in embedding:
            name_token = [label_uscored_uncased]
        else:
            name_token = [n for n in label if n in embedding]

        desc_tokens = [n for n in description if n in embedding]
        return name_token or ["_UNK_"], desc_tokens or ["_UNK_"]


def convert_entities(e: str, entity_info: Optional[Dict[str, str]] = None) -> Entity:
    if entity_info is None:
        return Entity(entity_id=e,
                      name='',
                      description='')
    else:
        return Entity(entity_id=e,
                      name=entity_info['label'],
                      description=entity_info['description'])


class Vocabulary:
    def __init__(self, entities: List[Entity], relations: List[str]):
        self.entities = entities
        self.relations = relations

        # Entity data
        self.key2entity: Dict[str, Entity] = {e.entity_id: e for e in self.entities}
        self.id2entity: Dict[int, Entity] = {i: e for i, e in enumerate(entities)}
        self.entity2id: Dict[Entity, id] = {e: i for i, e in enumerate(entities)}
        self.id2relation: Dict[int, str] = {i: r for i, r in enumerate(relations)}
        self.relation2id: Dict[str, int] = {r: i for i, r in enumerate(relations)}

        # Textual data, will be set after load_vectors is called
        self.vectors: torch.tensor = None
        self.token2id: DefaultDict[str, int] = None
        self.id2token: Dict[int, str] = None
        self.specials = ["_PAD_", "_UNK_", "_DESCSTART_", "_DESCEND_"]  # Make sure _PAD_ is always at 0

    def load_vectors(self, vectors: KeyedVectors, unknown_init: str = "rand") -> None:

        """
        :param vectors:
        :param unknown_init: one of "rand", "avg", "zero"
        :return:
        """
        logger.info('Loading word vectors for entities...')

        tokens = set()
        for entity in self.entities:
            entity.init_tokens(vectors)
            for t in entity.name_tokens + entity.description_tokens:
                if t in vectors and t not in self.specials:
                    tokens.add(t)

        AugmentationStatistics.log()  # log statistics about matched names in embedding

        self.id2token = {i: t for i, t in enumerate(self.specials + list(sorted(tokens)))}
        self.token2id = {t: i for i, t in self.id2token.items()}
        self.vectors = torch.zeros(len(self.id2token), vectors.vector_size)
        logger.info(f"Created word embedding with shape: {self.vectors.shape}")

        for i, t in self.id2token.items():
            self.token2id[t] = i

            if t not in self.specials:
                self.vectors[i] = torch.tensor(vectors[t])

        # initialize specials
        if unknown_init == "rand":
            self.vectors[self.token2id["_UNK_"]] = torch.randn(vectors.vector_size).normal_(mean=1., std=0.01)
        elif unknown_init == "avg":
            self.vectors[self.token2id["_UNK_"]] = torch.mean(self.vectors, dim=0)
        elif unknown_init == "zero":
            pass  # Already initialized with 0
        else:
            logger.warning("Invalid unknown token initialization value passed, will be initialized with 0.")

        self.vectors[self.token2id["_DESCSTART_"]] = torch.randn(vectors.vector_size).normal_(mean=1., std=0.01)
        self.vectors[self.token2id["_DESCEND_"]] = torch.randn(vectors.vector_size).normal_(mean=1., std=0.01)

        self._convert_entity_label_and_description()

    def _convert_entity_label_and_description(self):
        """
        Can be called after vectors and vocabulary have been loaded. Will numericalize label and description
        into a torch.tensor, that is saved in the entity.
        """
        if self.token2id is None or self.id2token is None or self.vectors is None:
            raise ValueError("This method does not work until vectors have been loaded.")

        for entity in self.entities:
            entity.name_tokens_idx = torch.tensor([self.token2id[t] for t in entity.name_tokens],
                                                  dtype=torch.long)
            entity.description_tokens_idx = torch.tensor([self.token2id[t] for t in entity.description_tokens],
                                                         dtype=torch.long)

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def num_tokens(self):
        return self.vectors.size(0)

    @property
    def vector_size(self):
        return self.vectors.size(1)


class PreprocessedTripleData:
    """
    A split of triple data.
    """

    def __init__(self, triples: List[Tuple[Entity, str, Entity]], vocab: Vocabulary):
        # self.triples = triples
        self.vocab = vocab
        self.heads, self.relations, self.tails = zip(*triples)
        heads, relations, tails = zip(*triples)

        self.heads_idx = torch.LongTensor([vocab.entity2id[entity] for entity in heads])
        self.relations_idx = torch.LongTensor([vocab.relation2id[relation] for relation in relations])
        self.tails_idx = torch.LongTensor([vocab.entity2id[entity] for entity in tails])

        unique_entities = {e.item() for e in torch.cat((self.heads_idx, self.tails_idx))}
        self.entities_idx: torch.tensor = torch.LongTensor(list(unique_entities))

        # will be set from RelationalDataset as it needs information from all splits
        self.head_labels: Dict[Tuple[int, int], Optional[torch.tensor]] = None  # { (tail1, relation) : [0,1,1,...,0], ..}
        self.tail_labels: Dict[Tuple[int, int], Optional[torch.tensor]] = None  # { (head1, relation) : [0,1,1,...,0], ..}

        self.rel2tails = self.get_rel2tails()  # target filtering
        self.heads_unique = set(self.heads_idx.numpy())
        self.tails_unique = set(self.tails_idx.numpy())

    def shuffle(self):
        """Shuffles this dataset"""

        perm = torch.randperm(len(self.heads_idx))
        self.heads_idx = self.heads_idx[perm]
        self.relations_idx = self.relations_idx[perm]
        self.tails_idx = self.tails_idx[perm]

        self.entities_idx = self.entities_idx[torch.randperm(len(self.entities_idx))]

    @property
    def num_entities(self) -> int:
        """ returns the number of entities in this split"""
        return len(self.entities_idx)

    @property
    def num_triples(self) -> int:
        return len(self.heads_idx)

    def iter_entitydata_triplewise(self, batch_size: int, yield_heads: bool = True,
                                   yield_tails: bool = True) -> Iterator[Tuple[torch.tensor,
                                                                               List[torch.tensor],
                                                                               List[torch.tensor]]]:
        """
        Yields entity, name tokens, and description tokens for an entity. Entities are taken out of the triples,
        which weights more frequent entities stronger. When yielding heads and tails this methods
        will alternating yield a batch of heads and then a batch of tails.
        """

        for batch_start in range(0, len(self.heads_idx), batch_size):
            batch_end = batch_start + batch_size

            if yield_heads:
                names = [self.vocab.id2entity[e.item()].name_tokens_idx for e in self.heads_idx[batch_start:batch_end]]
                descs = [self.vocab.id2entity[e.item()].description_tokens_idx for e in self.heads_idx[batch_start:batch_end]]
                yield self.heads_idx[batch_start:batch_end], names, descs
            if yield_tails:
                names = [self.vocab.id2entity[e.item()].name_tokens_idx for e in self.tails_idx[batch_start:batch_end]]
                descs = [self.vocab.id2entity[e.item()].description_tokens_idx for e in self.tails_idx[batch_start:batch_end]]
                yield self.tails_idx[batch_start:batch_end], names, descs

    def iter_entitydata_entitywise(self, batch_size: int, yield_heads: bool = True,
                                   yield_tails: bool = True) -> Iterator[Tuple[torch.tensor,
                                                                            List[torch.tensor],
                                                                            List[torch.tensor]]]:
        """
        Yields entity, name tokens, and description tokens for an entity.
        """
        for batch_start in range(0, len(self.entities_idx), batch_size):
            batch_end = batch_start + batch_size
            names = [self.vocab.id2entity[e.item()].name_tokens_idx for e in self.entities_idx[batch_start:batch_end]]
            descs = [self.vocab.id2entity[e.item()].description_tokens_idx for e in self.entities_idx[batch_start:batch_end]]
            yield self.entities_idx[batch_start:batch_end], names, descs

    def iter_triples(self, batch_size: int) -> Iterator[Tuple[torch.tensor, torch.tensor, torch.tensor,
                                                              torch.tensor, torch.tensor]]:
        """
        Iters the triples and yields and yields:
            heads, relations, tails and labels
        :param batch_size:
        :param shuffle: Shuffles dataset after each iteration.
        :param embedding: Text embedding used to return the embeddings in addition to indices
        """
        for batch_start in range(0, len(self.heads_idx), batch_size):
            heads_in_batch = self.heads_idx[batch_start:(batch_start + batch_size)]
            relations_in_batch = self.relations_idx[batch_start:(batch_start + batch_size)]
            tails_in_batch = self.tails_idx[batch_start:(batch_start + batch_size)]

            tail_labels_in_batch = [self.tail_labels[(h.item(), r.item())] for h, r in zip(heads_in_batch,
                                                                                           relations_in_batch)]
            head_labels_in_batch = [self.head_labels[(t.item(), r.item())] for t, r in zip(tails_in_batch,
                                                                                           relations_in_batch)]

            try:
                tail_labels_in_batch = torch.stack(tail_labels_in_batch)
            except TypeError:  # When any tail label is None, we're doing open-world head prediction (tails are unknown)
                tail_labels_in_batch = None

            try:
                head_labels_in_batch = torch.stack(head_labels_in_batch)
            except TypeError:  # When any head label is None, we're doing open-world tail prediction (heads are unknown)
                head_labels_in_batch = None

            yield heads_in_batch, relations_in_batch, tails_in_batch, head_labels_in_batch, tail_labels_in_batch

    def get_rel2tails(self):
        """
        Constructs a disctionary { (relation) : {tail1, tail2, tail3} }

        :return:
        """
        rel2tails = {}
        for head, relation, tail in zip(self.heads_idx, self.relations_idx, self.tails_idx):
            head, relation, tail = head.item(), relation.item(), tail.item()
            if not relation in rel2tails:
                rel2tails[relation] = set()
            rel2tails[relation].add(tail)
        return rel2tails


class RelationalDataset:

    def __init__(self, train_triples, validation_triples, test_triples, vocab, num_train_entities):

        self.train_triples = train_triples
        self.validation_triples = validation_triples
        self.test_triples = test_triples

        self.vocab = vocab

        self.train = PreprocessedTripleData(train_triples, self.vocab)
        self.validation = PreprocessedTripleData(validation_triples, self.vocab)
        self.test = PreprocessedTripleData(test_triples, self.vocab)

        # will be initialized with init_labels()
        self.head_labels: Dict[Tuple[int, int], Optional[torch.tensor]] = None  # { (tail1, relation) : [0,1,1,...,0], ..}
        self.tail_labels: Dict[Tuple[int, int], Optional[torch.tensor]] = None  # { (head1, relation) : [0,1,1,...,0], ..}
        self.init_labels()
        self.train.head_labels, self.train.tail_labels = self.head_labels, self.tail_labels
        self.validation.head_labels, self.validation.tail_labels = self.head_labels, self.tail_labels
        self.test.head_labels, self.test.tail_labels = self.head_labels, self.tail_labels


    def init_labels(self) -> None:
        heads_idx = torch.cat((self.train.heads_idx,
                               self.validation.heads_idx,
                               self.test.heads_idx))
        relations_idx = torch.cat((self.train.relations_idx,
                                   self.validation.relations_idx,
                                   self.test.relations_idx))
        tails_idx = torch.cat((self.train.tails_idx,
                               self.validation.tails_idx,
                               self.test.tails_idx))

        all_heads: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)  # { (tail1, relation) : {head1, ..} }
        all_tails: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)  # { (head1, relation) : {tail1, ..} }

        for head, relation, tail in zip(heads_idx, relations_idx, tails_idx):
            head, relation, tail = head.item(), relation.item(), tail.item()
            all_tails[(head, relation)].add(tail)
            all_heads[(tail, relation)].add(head)

        def to_one_hot(targets: Iterable, total: int) -> torch.tensor:
            """ Converts an iterable with indices to a one-hot representation."""
            for t in targets:
                if t >= total:
                    return  # We can't do this for unknown entities

            targets = torch.tensor(list(targets))
            labels = torch.zeros(total)
            labels[targets] = 1
            return labels

        self.head_labels = {k: to_one_hot(t, self.train.num_entities) for k, t in all_heads.items()}
        self.tail_labels = {k: to_one_hot(t, self.train.num_entities) for k, t in all_tails.items()}


def load_dataset(train_file: Union[str, pl.Path] = 'train.txt', valid_file: Union[str, pl.Path] = 'valid.txt',
                 test_file: Union[str, pl.Path] = 'test.txt', header: bool = False, split_symbol: str = '\t',
                 entitydata_file: Optional[Union[str, pl.Path]] = None) -> RelationalDataset:
    def read_entitydata(data_file: pl.Path) -> Dict:
        with data_file.open('rt') as f:
            return json.load(f)

    train_file = pl.Path(train_file)
    valid_file = pl.Path(valid_file)
    test_file = pl.Path(test_file)

    train_triples = load_triple_file(str(train_file), split_symbol=split_symbol, skip_header=header)
    valid_triples = load_triple_file(str(valid_file), split_symbol=split_symbol, skip_header=header)
    test_triples = load_triple_file(str(test_file), split_symbol=split_symbol, skip_header=header)

    # now convert entities
    def all_entities_and_relations(train_triples, validation_triples, test_triples):
        def get_distinct_entities(triples):
            s1, r, s2 = zip(*triples)
            s1 = set(s1)
            s2 = set(s2)
            return s1 | s2, set(r)

        distinct_train_entities, distinct_train_relations = get_distinct_entities(
            train_triples)
        logger.info("%s distinct entities in train having %s relations (%s triples)." % (
            len(distinct_train_entities),
            len(distinct_train_relations),
            len(train_triples)))

        distinct_valid_entities, distinct_valid_relations = get_distinct_entities(
            validation_triples)
        logger.info(
            "%s distinct entities in validation having %s relations (%s triples)." % (
                len(distinct_valid_entities),
                len(distinct_valid_relations),
                len(validation_triples)))

        distinct_test_entities, distinct_test_relations = get_distinct_entities(
            test_triples)
        logger.info("%s distinct entities in test having %s relations (%s triples)." % (
            len(distinct_test_entities),
            len(distinct_test_relations),
            len(test_triples)))

        # Create entity index, starting with the train entities
        all_entities = sorted(list(distinct_train_entities))
        # and then append the unknown ones from valid and test set
        all_entities += sorted(list((distinct_valid_entities | distinct_test_entities) - distinct_train_entities))

        all_relations = distinct_train_relations | distinct_valid_relations | distinct_test_relations
        all_relations = sorted(list(all_relations))
        logger.info("Working with: %s distinct entities having %s relations." % (
            len(all_entities), len(all_relations)))

        return all_entities, all_relations, len(distinct_train_entities)

    entities, relations, num_train_entities = all_entities_and_relations(train_triples, valid_triples, test_triples)

    num_entities = len(entities)
    logger.info("Converting entities...")
    with mp.Pool(mp.cpu_count() - 2) as p:
        if entitydata_file is not None:
            entitydata = read_entitydata(pl.Path(entitydata_file))
            entities = p.starmap(convert_entities, zip(entities, (entitydata[e] for e in entities)))
        else:
            entities = p.starmap(convert_entities, zip(entities, (None for _ in entities)))
    assert len(entities) == num_entities

    logger.info("Building Vocab...")
    vocab = Vocabulary(entities, relations)
    logger.info("Building triples...")
    train_triples = [(vocab.key2entity[k1], r, vocab.key2entity[k2])
                     for (k1, r, k2) in train_triples]
    valid_triples = [(vocab.key2entity[k1], r, vocab.key2entity[k2])
                     for (k1, r, k2) in valid_triples]
    test_triples = [(vocab.key2entity[k1], r, vocab.key2entity[k2])
                    for (k1, r, k2) in test_triples]

    return RelationalDataset(train_triples, valid_triples, test_triples, vocab, num_train_entities)


if __name__ == '__main__':
    dataset = load_dataset(
        train_file="../data/FB15k-237-zeroshot/train.txt",
        valid_file="../data/FB15k-237-zeroshot/valid_zero.txt",
        test_file="../data/FB15k-237-zeroshot/test_zero.txt",
        entitydata_file="../data/FB15k-237-zeroshot/entity2wikidata.json",

        # train_file="../data/FB15k-237-zeroshot/train.txt",
    )
    from IPython import embed;

    embed()
    vecs = load_embedding_file("/data/dok/johannes/pretrained_embeddings/wikipedia2vec/enwiki_20180420_300d.bin")
    dataset.vocab.load_vectors(vecs)
    from IPython import embed;

    embed()
