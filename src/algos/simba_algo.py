import torch
import torch.nn as nn
from torchvision import models
from abc import abstractmethod


class SimbaDefence(nn.Module):
    def __init__(self, utils):
        super(SimbaDefence, self).__init__()
        self.utils = utils

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def log_metrics(self):
        pass

    def train(self):
        self.mode = "train"
        self.client_model.train()

    def eval(self):
        self.mode = "val"
        self.client_model.eval()

    def init_client_model(self, config):
        if config["model_name"] == "resnet18":
            model = models.resnet18(pretrained=config["pretrained"])
            model = nn.Sequential(*nn.ModuleList(list(model.children())[:config["split_layer"]]))
        else:
            print("can't find client model")
            exit()

        return model

    def init_optim(self, config, model):
        if config["optimizer"] == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
        else:
            print("Unknown optimizer {}".format(config["optimizer"]))

        return optim

    def put_on_gpus(self):
        self.client_model = self.utils.model_on_gpus(self.client_model)

    def infer(self,data,labels):
        pass


class SimbaAttack(nn.Module):
    def __init__(self, utils):
        super(SimbaAttack, self).__init__()
        self.utils = utils

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    def train(self):
        self.mode = "train"
        self.model.train()

    def eval(self):
        self.mode = "val"
        self.model.eval()

    def init_optim(self, config, model):
        if config["optimizer"] == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
        else:
            print("Unknown optimizer {}".format(config["optimizer"]))

        return optim

    def train(self):
        self.mode = "train"
        self.model.train()

    def eval(self):
        self.mode = "val"
        self.model.eval()
