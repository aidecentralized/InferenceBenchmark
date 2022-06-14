import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from models.complex_models import ResNetProcessorComplex
from models.Unet import StochasticUNet
from models.Xception import Xception


def get_resnet18(logits):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Flatten(),
                             nn.Linear(num_ftrs, logits))
    return model

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
            
            pretrained = config["pretrained"] # this was repeated similarly in the main code - is that needed (Rohan)
            self.split_layer = config["split_layer"]
            if config["model_name"] == "resnet18":
                model = models.resnet18(pretrained=pretrained)
                
            elif config["model_name"] == "resnet34":
                model = models.resnet34(pretrained=pretrained)

            elif config["model_name"] == "resnet50":
                model = models.resnet50(pretrained=pretrained)

            elif config['model_name'] == "vgg16":
                model = models.vgg16(pretrained=pretrained)
            elif config['model_name'] == "xception":
                model = Xception(logits)
                self.model = model
            else:
                pass  

            if config["model_name"].startswith("resnet"):
                if config["model_name"] == "resnet20complex":
                    self.model = ResNetProcessorComplex(3,'alpha')
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, logits))
                model = nn.ModuleList(list(model.children())[self.split_layer:])
                self.model = nn.Sequential(*model)


            if config["model_name"].startswith("vgg"):
                num_ftrs = model.classifier[0].in_features
                model.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(num_ftrs, logits))
                model = nn.ModuleList(list(model.children())[self.split_layer:])
                self.model = nn.Sequential(*model)

        self.model = self.utils.model_on_gpus(self.model)
        self.utils.register_model("server_model", self.model)

    def assign_optim(self, config):
        lr = config["lr"]
        if config["optimizer"] == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self, z):
        self.z = z
        self.z.retain_grad()
        if self.config["model_name"] == "resnet20complex":
            self.h = self.model(self.z)
            h = self.h.detach()
            h.requires_grad = True
            return h
        else:
            x = self.model(self.z)
            self.preds = nn.functional.softmax(x, dim=1)
            return self.preds

    def compute_loss(self, preds, y):
        if self.config["model_name"] != "resnet20complex":
            self.loss = self.loss_fn(preds, y)
            self.utils.logger.add_entry(self.mode + "/" + self.loss_tag,
                                        self.loss.item())
            self.utils.logger.add_entry(self.mode + "/" + self.acc_tag,
                                        (preds, y), "acc")

    def optimize(self):
        if self.config["model_name"] != "resnet20complex":
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()
        else:
            self.optim.zero_grad()
            self.h.backward(self.decoder_grads)
            self.optim.step()

    def backward(self,y,decoder_grads=None):
        if decoder_grads != None:
            self.decoder_grads = decoder_grads
            self.optimize()
        if self.config["model_name"] != "resnet20complex":
            self.compute_loss(self.preds, y)
            self.optimize()
        return self.z.grad

    def processing(self, z, y):
        z.retain_grad()
        preds = self.forward(z)
        self.compute_loss(preds, y)
        self.optimize()
        return z.grad
