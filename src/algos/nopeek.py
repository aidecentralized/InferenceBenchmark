from algos.simba_algo import SimbaDefence
import torch
from torch.nn.modules.loss import _Loss
import numpy as np


class DistCorrelation(_Loss):
    def __init__(self, ):
        super(DistCorrelation, self).__init__()

    def pairwise_distances(self, x):
        '''Taken from: https://discuss.pytorch.org/t/batched-pairwise-distance/39611'''
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist[dist != dist] = 0  # replace nan values with 0
        return torch.clamp(dist, 0.0, np.inf)

    def forward(self, z, data):
        z = z.reshape(z.shape[0], -1)
        data = data.reshape(data.shape[0], -1)
        a = self.pairwise_distances(z)
        b = self.pairwise_distances(data)
        a_centered = a - a.mean(dim=0).unsqueeze(1) - a.mean(dim=1) + a.mean()
        b_centered = b - b.mean(dim=0).unsqueeze(1) - b.mean(dim=1) + b.mean()
        dCOVab = torch.sqrt(torch.sum(a_centered * b_centered) / a.shape[1]**2)
        var_aa = torch.sqrt(torch.sum(a_centered * a_centered) / a.shape[1]**2)
        var_bb = torch.sqrt(torch.sum(b_centered * b_centered) / a.shape[1]**2)

        dCORab = dCOVab / torch.sqrt(var_aa * var_bb)
        return dCORab


class NoPeek(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(NoPeek, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)
        self.loss = DistCorrelation()

        self.alpha = config["alpha"]
        self.dcor_tag = "dcor"
        self.utils.logger.register_tag("train/" + self.dcor_tag)
        self.utils.logger.register_tag("val/" + self.dcor_tag)

    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)
        self.x = x
        z = self.z.detach()
        z.requires_grad = True
        self.dcor_loss = self.loss(self.x, self.z)
        self.utils.logger.add_entry(self.mode + "/" + self.dcor_tag,
                                    self.dcor_loss.item())
        return z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        # Higher the alpha, higher the weight for dcor loss would be
        self.z.backward((1 - self.alpha) * server_grads, retain_graph=True)
        (self.alpha * self.dcor_loss).backward()
        self.optim.step()
