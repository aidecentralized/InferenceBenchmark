import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from models.complex_models import ResNetProcessorComplex

class Model(nn.Module):
    def __init__(self, config, utils):
        super(Model, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.utils = utils
        if config["model_name"] != "resnet20complex":
            self.loss_tag = "server_loss"
            self.acc_tag = "server_acc"
            self.utils.logger.register_tag("train/" + self.loss_tag)
            self.utils.logger.register_tag("val/" + self.loss_tag)
            self.utils.logger.register_tag("train/" + self.acc_tag)
            self.utils.logger.register_tag("val/" + self.acc_tag)
        self.config = config
        self.assign_model(config)
        self.assign_optim(config)

    def train(self):
        self.mode = "train"
        self.model.train()

    def eval(self):
        self.mode = "val"
        self.model.eval()

    def assign_model(self, config):
        logits = config["logits"]
        if config["model_name"] == "feed_forward":
            layer_list = []
            for l in range(config["num_layers"] - 1):
                layer_list.append(nn.Linear(logits[l], logits[l+1]))
            layer_list.append(nn.Linear(logits[l], logits[l+1]))
            self.model = nn.Sequential(*nn.ModuleList(layer_list))
        else:
            pretrained = config["pretrained"]
            self.split_layer = config["split_layer"]
            if config["model_name"] == "resnet18":
                pretrained = config["pretrained"]
                model = models.resnet18(pretrained=pretrained)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, logits))
                model = nn.ModuleList(list(model.children())[self.split_layer:])
                self.model = nn.Sequential(*model)
            elif config["model_name"] == "resnet20complex":
                self.model = ResNetProcessorComplex(3,'alpha')
        self.model = self.utils.model_on_gpus(self.model)
        self.utils.register_model("server_model", self.model)

    def assign_optim(self, config):
        lr = config["lr"]
        if config["model_name"] == "resnet20complex":
            
            if config["optimizer"] == "adam":
                self.optim = torch.optim.Adam(
                    self.model.parameters(),
                    lr,
                )

            elif config["optimizer"] == "sgd":
                self.optim = torch.optim.SGD(
                    self.model.parameters(),
                    lr,
                    momentum = config["momentum"],
                    weight_decay = config["weight_decay"]
                )
        else:
            if config["optimizer"] == "adam":
                self.optim = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self, z):
        self.z = z
        self.z.retain_grad()
        if self.config["model_name"] == "resnet20complex":
            h = self.model(self.z)
            assert(h.size(1) == 2)
            self.preds = None
            h = h.detach()
            h.requires_grad = True
            return h
        else:
            x = self.model(self.z)
            self.preds = nn.functional.softmax(x, dim=1)
            return None

    def compute_loss(self, preds, y):
        if self.config["model_name"] != "resnet20complex":
            self.loss = self.loss_fn(preds, y)
            self.utils.logger.add_entry(self.mode + "/" + self.loss_tag,
                                        self.loss.item())
            self.utils.logger.add_entry(self.mode + "/" + self.acc_tag,
                                        (preds, y), "acc")
    def optimize(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    def backward(self,y,decoder_loss):
        self.loss = decoder_loss
        self.compute_loss(self.preds, y)
        self.optimize()
        return self.z.grad

