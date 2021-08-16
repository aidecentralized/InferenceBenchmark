import torch
import torchvision.models as models
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config, utils):
        super(Model, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.split_layer = config["split_layer"]
        self.utils = utils
        self.assign_model(config)
        self.assign_optim(config)

    def assign_model(self, config):
        pretrained = config["pretrained"]
        logits = config["logits"]
        if config["model_name"] == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Flatten(),
                                     nn.Linear(num_ftrs, logits))
            model = nn.ModuleList(list(model.children())[self.split_layer:])
            self.model = nn.Sequential(*model)
            self.model = self.utils.model_on_gpus(self.model)

    def assign_optim(self, config):
        lr = config["lr"]
        if config["optimizer"] == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr)

    def forward(self, x):
        x = self.model(x)
        """for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)"""
        return nn.functional.softmax(x, dim=1)

    def compute_loss(self, preds, y):
        self.loss = self.loss_fn(preds, y)
        print(self.loss)

    def optimize(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    def processing(self, z, y):
        z.retain_grad()
        preds = self.forward(z)
        self.compute_loss(preds, y)
        self.optimize()
        return z.grad
