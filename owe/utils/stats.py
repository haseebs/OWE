import logging

#from owe.utils import plots

logger = logging.getLogger("owe")


class Statistics:
    def __init__(self, outdir, writer):
        self.outdir = outdir
        self.writer = writer
        self.train_losses = []
        self.train_losses_i = []
        self.validation_losses = []
        self.hitsat1 = []
        self.hitsat3 = []
        self.hitsat10 = []
        self.mean_rec_rank_filtered = []
        self.mean_rec_rank_raw = []
        self.mean_rank = []
        self.mean_rank_raw = []
        self.metric_epochs = []
        self.nn_mean_rank = []
        self.nn_mean_rank_i = []
        self.nn_mrr = []
        self.nn_mrr_i = []

    def record(self, train_losses=None,  # TODO add type hints
               train_losses_i=None,
               validation_losses=None,
               hitsat1=None,
               hitsat3=None,
               hitsat10=None,
               mean_rec_rank_filtered=None,
               mean_rec_rank_raw=None,
               mean_rank=None,
               mean_rank_raw=None,
               metric_epochs=None,
               nn_mean_rank=None,
               nn_mean_rank_i=None,
               nn_mrr=None,
               nn_mrr_i=None):
        """
        Stores the statistics in the respective lists.
        """
        self.train_losses.append(train_losses) if train_losses is not None else {}
        self.train_losses_i.append(train_losses_i) if train_losses_i is not None else {}
        self.validation_losses.append(validation_losses) if validation_losses is not None else {}
        self.hitsat1.append(hitsat1) if hitsat1 is not None else {}
        self.hitsat3.append(hitsat3) if hitsat3 is not None else {}
        self.hitsat10.append(hitsat10) if hitsat10 is not None else {}
        self.mean_rec_rank_filtered.append(mean_rec_rank_filtered) if mean_rec_rank_filtered is not None else {}
        self.mean_rec_rank_raw.append(mean_rec_rank_raw) if mean_rec_rank_raw is not None else {}
        self.mean_rank.append(mean_rank) if mean_rank is not None else {}
        self.mean_rank_raw.append(mean_rank_raw) if mean_rank_raw is not None else {}
        self.metric_epochs.append(metric_epochs) if metric_epochs is not None else {}
        self.nn_mean_rank.append(nn_mean_rank) if nn_mean_rank is not None else {}
        self.nn_mean_rank_i.append(nn_mean_rank_i) if nn_mean_rank_i is not None else {}
        self.nn_mrr.append(nn_mrr) if nn_mrr is not None else {}
        self.nn_mrr_i.append(nn_mrr_i) if nn_mrr_i is not None else {}

    def log_losses(self, epoch):
        """
        Output the loss on the stdout.
        :param epoch: current epoch:
        :return:
        """
        logger.info("At epoch {}. Train Loss: {} ".format(epoch, self.train_losses[-1]))

    def log_eval(self, epoch, dataset_name):
        """
        Output the evaluation metrics to stdout.
        :param epoch: current epoch
        :param dataset_name: test or validation
        :return:
        """
        logger.info("[Eval: {}] Epoch: {}".format(dataset_name, epoch))
        logger.info("[Eval: {}] {:6.2f} Hits@1 (%)".format(dataset_name, self.hitsat1[-1] * 100))
        logger.info("[Eval: {}] {:6.2f} Hits@3 (%)".format(dataset_name, self.hitsat3[-1] * 100))
        logger.info("[Eval: {}] {:6.2f} Hits@10 (%)".format(dataset_name, self.hitsat10[-1] * 100))
        logger.info("[Eval: {}] {:6.2f} MRR (filtered) (%)".format(dataset_name, self.mean_rec_rank_filtered[-1] * 100))
        logger.info("[Eval: {}] {:6.2f} MRR (raw) (%)".format(dataset_name, self.mean_rec_rank_raw[-1] * 100))
        logger.info("[Eval: {}] Mean rank: {}".format(dataset_name, self.mean_rank[-1]))
        logger.info("[Eval: {}] Mean rank raw: {}".format(dataset_name, self.mean_rank_raw[-1]))

    def log_nn_ranks(self, epoch):
        """
        Output nearest neighbour mean ranks to stdout.
        Only implemented for complEx yet.
        :param epoch: current epoch
        :return:
        """
        if not self.nn_mean_rank_i:
            raise NotImplementedError
        logger.info("At epoch {}. Train nn mean rank real: {} "
                    "Train nn mean rank img: {}"
                    "Train nn mrr: {} Train nn mrr img: {}".format(epoch,
                                                                   self.nn_mean_rank[-1],
                                                                   self.nn_mean_rank_i[-1],
                                                                   self.nn_mrr[-1],
                                                                   self.nn_mrr_i[-1]))

    def push_losses(self, epoch, verbose=True):
        """
        Output losses to tensorboard.
        :param epoch: current epoch
        :param verbose: whether the losses should be printed to stdout too
        :return:
        """
        self.writer.add_scalar('losses/train', self.train_losses[-1], epoch)
        if self.validation_losses:
            self.writer.add_scalar('losses/validation', self.validation_losses[-1], epoch)
        if self.train_losses_i:
            self.writer.add_scalar('losses/train_img', self.train_losses_i[-1], epoch)
        if verbose:
            self.log_losses(epoch)

    def push_eval(self, epoch, dataset_name, verbose=True):
        """
        Output evaluation metrics to tensorboard.
        :param epoch: current epoch
        :param dataset_name: test or validation
        :param verbose: whether the metrics should be printed to stdout too
        :return:
        """
        group = "test/"
        if dataset_name == "validation":
            group = "eval/"
        self.writer.add_scalar(group + 'Hits@1 %', self.hitsat1[-1] * 100, epoch)
        self.writer.add_scalar(group + 'Hits@3 %', self.hitsat3[-1] * 100, epoch)
        self.writer.add_scalar(group + 'Hits@10 %', self.hitsat10[-1] * 100, epoch)
        self.writer.add_scalar(group + 'MRR (filtered) %', self.mean_rec_rank_filtered[-1] * 100, epoch)
        self.writer.add_scalar(group + 'MRR (raw) %', self.mean_rec_rank_raw[-1] * 100, epoch)
        self.writer.add_scalar(group + 'Mean rank', self.mean_rank[-1], epoch)
        self.writer.add_scalar(group + 'Mean rank raw', self.mean_rank_raw[-1], epoch)
        if verbose:
            self.log_eval(epoch, dataset_name)

    def push_nn_ranks(self, epoch, verbose=True):
        """
        Output nearest neighbour mean ranks to tensorboard.
        :param epoch: current epoch
        :param verbose: whether the ranks should be printed to stdout too.
        """
        self.writer.add_scalar('NN/Mean rank real', self.nn_mean_rank[-1], epoch)
        self.writer.add_scalar('NN/Mean rank img', self.nn_mean_rank_i[-1], epoch)
        self.writer.add_scalar('NN/MRR', self.nn_mrr[-1], epoch)
        self.writer.add_scalar('NN/MRR img', self.nn_mrr_i[-1], epoch)
        if verbose:
            self.log_nn_ranks(epoch)

    def plot_metrics(self):
        """
        Output the metrics as a matplotlib figure.
        :return:
        """
        plots.plot_curves(filename=str(self.outdir / 'metrics.png'),
                          curves=(self.mean_rec_rank_filtered,
                                  self.mean_rec_rank_raw,
                                  self.hitsat1,
                                  self.hitsat3,
                                  self.hitsat10),
                          labels=("MRR (filtered)", "MRR (raw)", "Hits@1",
                                  "Hits@3", "Hits@10"),
                          x_label='Epoch', y_label='%', x=metric_epochs)  # FIXME: metric_epochs unknown

    def plot_losses(self):
        """
        Output the losses as a matplotlib figure.
        :return:
        """
        plots.plot_curves(filename=str(self.outdir / 'losses.png'),
                          curves=(self.train_losses,
                                  self.train_losses_i),
                          labels=('Train real', 'Train img'),
                          x_label='Epoch', y_label='Loss')
