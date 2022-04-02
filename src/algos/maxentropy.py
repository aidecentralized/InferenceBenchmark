from algos.simba_algo import SimbaDefence
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn.modules.loss import _Loss
torch.autograd.set_detect_anomaly(True)
distance = nn.CrossEntropyLoss()
from algos.deepobfuscator import DeepObfuscator
from utils.metrics import MetricLoader

class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        if (input < 0).any() or (input > 1).any():
            print(input)
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(input * torch.log(input), dim=0))

        return H

class MaxEntropy(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(MaxEntropy, self).__init__(utils)
        self.initialize(config, utils.device)

    def initialize(self, config, device):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.client_optim = self.init_optim(config, self.client_model)
        self.entropy_loss_fn = EntropyLoss()

    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, items):
        entropy_loss = self.entropy_loss_fn(items["pred_lbls"])
        entropy_loss.requires_grad = True
        entropy_loss.backward()
        self.z.backward(items["server_grads"])
        self.client_optim.step()
        H = torch.mean(torch.sum(input * torch.log(input), dim=1))

        return H



class maxentropy(DeepObfuscator):
    """ The only difference between maxentropy and Deepobfuscator
    is the loss function for the proxy adversary and the label is the private attribute instead of reconstruction.
    """
    def __init__(self, config, utils) -> None:
        super(maxentropy, self).__init__(config, utils)
        self.update_loss()

    def update_loss(self):
        self.loss = EntropyLoss().forward

    def get_adv_loss(self):
        # Since it is L1, it has to be minimized
        return self.adv_loss
