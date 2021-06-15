import torch
import torch.nn as nn
from torchvision import models

class AdversaryModelPred(nn.Module):
    """ Simple adversarial predictive model"""
    def __init__(self, split_layer, logits=7, pretrained=False):
        super(AdversaryModelPred, self).__init__()
        self.logits = logits
        self.split_layer = split_layer

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer: continue
            x = l(x)
        return nn.functional.softmax(x, dim=1)
