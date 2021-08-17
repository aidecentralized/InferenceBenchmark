import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from utils.logs import Logs
from shutil import copytree, copy2
from glob import glob
import shutil


class Utils():
    def __init__(self, config) -> None:
        self.config = config
        self.model_registry = {}

        self.gpu_devices = config.get("gpu_devices")
        gpu_id = self.gpu_devices[0]

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

    def init_logger(self):
        self.logger = Logs(self.config)

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

        # For saving models in the future
        os.mkdir(self.config.get('model_path'))

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
            os.makedirs(path)
        else:
            print("Input not understood")
            exit()
        """elif inp == "c":
            experiment_path = path"""
        """if not experiment_path:
            self.copy_source_code(path)
            os.mkdir(self.config.get('model_path'))
            os.mkdir(self.config.get('log_path'))"""

    def register_model(self, key, model):
        if self.model_registry.get(key) is None:
            self.model_registry[key] = model
        else:
            self.logger.log_console("model {} is already registered".format(key))

    def _save_model(self, state_dict, path):
        torch.save(state_dict, path)

    def save_models(self):
        for model_name, model in self.model_registry.items():
            model_path = self.config["model_path"] + "/{}.pt".format(model_name)
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            self._save_model(state_dict, model_path)
        self.logger.log_console("models saved")