from typing import Optional

from owe.models import ComplEx, DistMult, TransE, RotatE


class KGCFactory:

    @classmethod
    def get_model(cls, model_name: str, num_entities: int, num_relations: int, embedding_dim: int,
                  rotate_gamma: Optional[float] = None):
        if model_name.lower() == "complex":
            return ComplEx(num_entities=num_entities,
                           num_relations=num_relations,
                           embedding_dim=embedding_dim)
        elif model_name.lower() == "transe":
            return TransE(num_entities=num_entities,
                          num_relations=num_relations,
                          embedding_dim=embedding_dim)
        elif model_name.lower() == "distmult":
            return DistMult(num_entities=num_entities,
                            num_relations=num_relations,
                            embedding_dim=embedding_dim)
        elif model_name.lower() == "rotate":
            return RotatE(num_entities=num_entities,
                          num_relations=num_relations,
                          embedding_dim=embedding_dim,
                          gamma=rotate_gamma)
        else:
            raise ValueError(f"LinkPredictionModelType '{model_name}' unknown")
