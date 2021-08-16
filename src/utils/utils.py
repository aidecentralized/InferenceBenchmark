import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import logging
from metrics.loaders import MetricLoader
from shutil import copytree, copy2
from glob import glob
import shutil


class Logs():
    def __init__(self, config):
        self.init_tb(config)
        self.init_logfile(config)
        self.keys = {}

    def register_key(self, key):
        if self.keys.get(key) is not None:
            print("duplicate key {}".format(key))
        self.keys[key] = []

    def init_logfile(self, config):
        self.log_path = config["log_path"]
        self.log_format = "%(asctime)s::%(levelname)s::%(name)s::"\
                          "%(filename)s::%(lineno)d::%(message)s"
        logging.basicConfig(filename="{log_path}/log_console.log".format(
                                                  log_path=self.log_path),
                            level='DEBUG', format=self.log_format)

    def init_tb(self, config):
        log_path = config["log_path"]
        tb_path = log_path + "/tensorboard"
        if not os.path.exists(tb_path) or not os.path.isdir(tb_path):
            os.mkdir(tb_path)
        self.writer = SummaryWriter(tb_path)

    def log_console(self, msg):
        logging.info(msg)

    def log_tb(self, key, value, iteration):
        self.writer.add_scalar(key, value, iteration)

    def log_data(self, key, data, metric, iteration):
        """ Metric is computed on data and logged with the key. If the metric is "precomputed"
        then the data is directly stored with the key """
        if metric == "acc":
            data = self.metrics.acc(data[0], data[1])
        if metric == "precomputed":
            pass
        self.log_console("iteration {}, {}: {}".format(iteration, key, data))
        self.log_tb(key, data, iteration)

class Utils():
    def __init__(self, config) -> None:
        self.config = config
        self.utils = Logs(config)
        self.metrics = MetricLoader()

        self.gpu_devices = config.get("gpu_devices")
        gpu_id = self.gpu_devices[0]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

    def model_on_gpus(self, model):
        total_gpus = len(self.gpu_devices)
        if total_gpus > 1:
            return nn.DataParallel(model.to(self.device), device_ids = self.gpu_devices)
        elif total_gpus == 1:
            return model.to(self.device)
        else:
            # Only CPU available
            return model

    def get_data(self, sample):
        data = Variable(sample["img"]).to(self.device)
        prediction_labels = Variable(sample["prediction_label"]).to(self.device)
        protected_labels = Variable(sample["private_label"]).to(self.device)
        return data, prediction_labels, protected_labels

    def copy_source_code(self, path):
        print("exp path:", path)
        if os.path.isdir(path):
            # throw a prompt
            self.check_path_status(path)
        else:
            os.makedirs(path)

        denylist = ["./__pycache__/"]
        folders = glob(r'./*/')

        # For copying python files
        for file_ in glob(r'./*.py'):
            copy2(file_, path)

        # For copying json files
        for file_ in glob(r'./*.json'):
            copy2(file_, path)

        for folder in folders:
            if folder not in denylist:
                # Remove first char which is . due to the glob
                copytree(folder, path + folder[1:])

    def check_path_status(self, path):
        """experiment_path = None
        if auto:  # This is to not duplicate work already done and to continue running experiments
            print("silently skipping experiment",
                  self.config.get('experiment_name'))
            return None"""
        print("Experiment {} already present".format(self.config.get("experiment_name")))
        inp = input("Press e to exit, r to replace it, c to continue training: ")
        if inp == "e":
            exit()
        elif inp == "r":
            shutil.rmtree(path)
        else:
            print("Input not understood")
            exit()
        """elif inp == "c":
            experiment_path = path"""
        """if not experiment_path:
            self.copy_source_code(path)
            os.mkdir(self.config.get('model_path'))
            os.mkdir(self.config.get('log_path'))"""
