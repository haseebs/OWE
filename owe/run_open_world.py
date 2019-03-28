from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
import logging
from pathlib import Path
from typing import Tuple, List
import subprocess

from tensorboardX import SummaryWriter

from owe import data
from owe.config import Config
from owe.models import KGCFactory
from owe.open_world_prediction import OpenWorldPrediction
from owe.utils import read_config, load_checkpoint

LOG_LEVEL = logging.INFO
logger = logging.getLogger("owe")


def setup_logging(level: int = logging.INFO, logfile: str = None) -> None:
    global logger
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', "%H:%M:%S")
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def parse_args() -> Tuple[Namespace, List[str]]:
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, description=globals()['__doc__'])
    parser.add_argument('-c', '--corpus_directory', help="Directory where the dataset is stored", required=True)
    parser.add_argument('-d', '--output_directory', help="Where to output artifacts.", required=True)
    parser.add_argument('-C', '--config', help="Path to config.ini. Default: config.ini inside output directory.",
                        required=False)

    # Pretrained model
    model_parser = parser.add_mutually_exclusive_group(required=True)
    model_parser.add_argument('--distmult', help="Load a pretrained DistMult model from given directory.")
    model_parser.add_argument('--transe', help="Load a pretrained TransE model from given directory.")
    model_parser.add_argument('--complex', help="Load a pretrained ComplEx model from given directory.")
    model_parser.add_argument('--rotate', help="Load a pretrained RotatE model from given directory.")
    parser.add_argument('--gamma', help="Specify the gamma value of the pretrained RotatE model.", required=False, type=float)

    # Run closed-world or open-world model
    mode_parser = parser.add_mutually_exclusive_group(required=True)
    mode_parser.add_argument('-v', '--validate', help="Evaluate the model on validation set.", action='store_true')
    mode_parser.add_argument('-e', '--evaluate', help="Evaluate the model on test set.", action='store_true')

    mode_parser.add_argument('-t', '--train', help="Train an OWE model on test set.", action='store_true')

    # Train OWE model
    checkpoint_parser = parser.add_mutually_exclusive_group()
    checkpoint_parser.add_argument('-l', '--load', help="Load the last OWE model checkpoint if one exists.",
                                   action='store_true')
    checkpoint_parser.add_argument('-lb', '--load_best', help="Load the best OWE model checkpoint if one exists.",
                                   action='store_true')
    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    corpus_directory = Path(args.corpus_directory)

    # Init experiment directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(str(output_directory))
    setup_logging(logfile=str(output_directory / 'output.log'))

    # Log git hash for reproducibility
    try:
        logger.info("Git Hash: " + subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8"))
    except subprocess.CalledProcessError:
        logger.info("Could not log git commit hash.")

    # Try to load config file
    config_file = Path(args.config) if args.config is not None else output_directory / "config.ini"
    if not config_file.exists():
        raise FileNotFoundError(f"No config file found under: {config_file}.")
    train_file, valid_file, test_file, skip_header, split_symbol, wiki_file = read_config(config_file)

    # Load dataset
    dataset = data.load_dataset(train_file=corpus_directory / train_file,
                                valid_file=corpus_directory / valid_file,
                                test_file=corpus_directory / test_file,
                                split_symbol=split_symbol,
                                header=skip_header,
                                entitydata_file=corpus_directory / wiki_file)

    # Load pretrained word embeddings
    word_vectors = data.load_embedding_file(Config.get('PretrainedEmbeddingFile'))
    embedding_dim = word_vectors.vector_size
    logger.info("Building embedding matrix")
    dataset.vocab.load_vectors(word_vectors)  # Create embedding for known words
    del word_vectors

    # load KGC model
    kgc_model_name, kgc_model_directory = None, None
    if args.complex:
        kgc_model_name, kgc_model_directory = "ComplEx", Path(args.complex)
    elif args.transe:
        kgc_model_name, kgc_model_directory = "TransE", Path(args.transe)
    elif args.distmult:
        kgc_model_name, kgc_model_directory = "DistMult", Path(args.distmult)
    elif args.rotate:
        kgc_model_name, kgc_model_directory = "RotatE", Path(args.rotate)
        if args.gamma is None:
            raise ValueError("Specify gamma value from trained RotatE")

    Config.set("LinkPredictionModelType", kgc_model_name)
    logger.info(f"LinkPredictionModelType: {kgc_model_name}")

    kgc_model = KGCFactory.get_model(kgc_model_name, dataset.train.num_entities,
                                     dataset.vocab.num_relations, embedding_dim, rotate_gamma=args.gamma)
    kgc_model.init_embeddings(dataset, kgc_model_directory)

    model = OpenWorldPrediction(kgc_model,
                                dataset,
                                embedding_dim=embedding_dim,
                                outdir=output_directory,
                                writer=tensorboard_writer)

    # Load OWE checkpoint (different from the KGC embeddings)
    start_epoch = 1

    if args.load or args.load_best:
        checkpoint_file = 'best_checkpoint.OWE.pth.tar' if args.load_best else 'checkpoint.OWE.pth.tar'
        checkpoint_owe = load_checkpoint(output_directory / checkpoint_file)

        if checkpoint_owe:
            start_epoch = checkpoint_owe['epoch'] + 1
            model.kgc_model.load_state_dict(checkpoint_owe['model'])
            model.mapper.load_state_dict(checkpoint_owe["mapper"])
            model.optimizer.load_state_dict(checkpoint_owe["optimizer"])
            logger.info("Initialized OWE model, mapper and optimizer from the loaded checkpoint.")

        del checkpoint_owe

    if args.evaluate:
        model.evaluate(preprocessed_data=model.dataset.test,
                       dataset_name='test',
                       print_predictions=False,
                       epoch=start_epoch - 1)
    elif args.validate:
        model.evaluate(preprocessed_data=model.dataset.validation,
                       dataset_name='validation',
                       print_predictions=False,
                       epoch=start_epoch - 1)
    else:
        model.train(epochs=Config.get("Epochs"),
                    batch_size=Config.get("BatchSize"),
                    validate_every=Config.get("ValidateEvery"))


if __name__ == '__main__':
    main()
