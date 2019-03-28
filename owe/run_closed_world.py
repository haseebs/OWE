import argparse
import logging
from pathlib import Path
import subprocess

from owe import data
from owe.models import KGCFactory
from owe.config import Config
from owe.closed_world_prediction import evaluate_closed_world

LOG_LEVEL = logging.DEBUG
logger = logging.getLogger("owe")


def setup_logging(level=logging.INFO):
    global logger
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', "%H:%M:%S")
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=globals()['__doc__'])
    parser.add_argument('-c', '--corpus_directory', help="Directory where the dataset is stored", required=True)
    parser.add_argument('-dim', '--embedding_dim', help="Embedding dimensionality.", required=True, type=int)
    parser.add_argument('-b', '--batch_size', help="Evaluation batch size.", required=True, type=int, default=16)

    # Pretrained model
    model_parser = parser.add_mutually_exclusive_group(required=True)
    model_parser.add_argument('--distmult', help="Load a pretrained DistMult model from given directory.")
    model_parser.add_argument('--transe', help="Load a pretrained TransE model from given directory.")
    model_parser.add_argument('--complex', help="Load a pretrained ComplEx model from given directory.")

    model_parser.add_argument('--rotate', help="Load a pretrained RotatE model from given directory.")
    parser.add_argument('--gamma', help="Specify the gamma value of the pretrained RotatE model.", required=False, type=float)

    # Run closed-world model
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument('-v', '--validate', help="Evaluate the model on validation set.", action='store_true')
    mode_parser.add_argument('-e', '--evaluate', help="Evaluate the model on test set.", action='store_true')

    args, unknown = parser.parse_known_args()
    return args, unknown


def basic_config():
    Config.set("device", "cuda")
    Config.set("ConvertEntities", False)
    Config.set("InitializeEmbeddingWithAllEntities", False)


def main():
    setup_logging(LOG_LEVEL)

    args, _ = parse_args()
    corpus_directory = Path(args.corpus_directory)
    basic_config()

    # Log git hash for reproducibility
    try:
        logger.info("Git Hash: " + subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8"))
    except subprocess.CalledProcessError:
        logger.info("Could not log git commit hash.")

    dataset = data.load_dataset(train_file=corpus_directory / "train.txt",
                                valid_file=corpus_directory / "valid.txt",
                                test_file=corpus_directory / "test.txt",
                                split_symbol="\t",
                                header=False,
                                entitydata_file=None)

    # load KGC model
    model_name, model_dir = None, None
    if args.complex:
        model_name, model_dir = "complex", Path(args.complex)
    elif args.transe:
        model_name, model_dir = "transe", Path(args.transe)
    elif args.distmult:
        model_name, model_dir = "distmult", Path(args.distmult)
    elif args.rotate:
        model_name, model_dir = "rotate", Path(args.rotate)
        if args.gamma is None:
            raise ValueError("Specify gamma value from trained RotatE")

    model = KGCFactory.get_model(model_name, dataset.train.num_entities, dataset.vocab.num_relations,
                                 args.embedding_dim, rotate_gamma=args.gamma)
    model = model.to("cuda")
    model.init_embeddings(dataset, model_dir)

    if args.evaluate:
        evaluate_closed_world(model,
                              preprocessed_data=dataset.test,
                              batch_size=args.batch_size,
                              dataset_name='test')
    if args.validate:
        evaluate_closed_world(model,
                              preprocessed_data=dataset.validation,
                              batch_size=args.batch_size,
                              dataset_name='validation')


if __name__ == '__main__':
    main()
