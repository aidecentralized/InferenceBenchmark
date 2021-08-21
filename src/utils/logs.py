from tensorboardX import SummaryWriter
import os
import logging
from utils.metrics import MetricLoader
import numpy as np


class Logs():
    def __init__(self, config, challenge=False):
        if not challenge:
            self.init_tb(config)
        self.init_logfile(config, challenge)
        self.epoch = 0
        self.curr_iters = 0
        self.total_epochs = config["total_epochs"]
        self.items = {}
        self.metrics = MetricLoader()

    def register_tag(self, key):
        if self.items.get(key) is not None:
            print("duplicate key {}".format(key))
        self.items[key] = []

    def set_log_freq(self, total_iters):
        log_freq = 10 # 10 times every epoch
        self.trigger_freq = total_iters // log_freq
        self.total_iters = total_iters

    def init_logfile(self, config, challenge):
        self.log_path = config["log_path"]
        self.log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                          "%(filename)s::%(lineno)d::%(message)s"
        if not challenge:
            logging.basicConfig(filename="{log_path}/log_console.log".format(
                                                     log_path=self.log_path),
                                level='DEBUG', format=self.log_format)
        logging.getLogger().addHandler(logging.StreamHandler())

    def init_tb(self, config):
        log_path = config["log_path"]
        tb_path = log_path + "/tensorboard"
        # if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
        os.makedirs(tb_path)
        self.writer = SummaryWriter(tb_path)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)

    def log_data(self, key, val, iteration):
        self.log_console("epoch {}, iteration {}, {}: {:.4f}".format(self.epoch,
                                                                     iteration, key, val))
        self.log_tb(key, val, iteration)

    def add_entry(self, key, data, metric=None):
        """ Metric is computed on data and logged with the key. If the metric is None
        then the data is directly stored with the key """
        if metric == "acc":
            val = self.metrics.acc(data[0], data[1])
        else:
            val = data
        self.items[key].append(val)
        self.curr_iters += 1
        # flush logs only if we are in validation mode
        if key.startswith("train"):
            # Since iteration is reset after every epoch
            iteration = self.epoch * self.total_iters + self.curr_iters
            self.log_tb(key, val, iteration)
            if self.curr_iters % self.trigger_freq == 0:
                self.log_data(key, val, self.curr_iters)

    def flush_epoch(self):
        self.epoch += 1
        for key, val in self.items.items():
            # no need for computing training statistic for now due to tensorboard problem of
            # confusion between epoch and iters
            if key.startswith("val"):
                avg_val = np.array(val).sum() / len(val)
                self.log_tb(key, avg_val, self.epoch)
                self.log_data(key, avg_val, self.epoch)
            self.items[key] = []
        self.curr_iters = 0