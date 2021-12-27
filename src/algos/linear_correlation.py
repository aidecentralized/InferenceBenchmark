from algos.simba_algo import SimbaDefence
from torch.nn.modules.loss import _Loss
from audtorch.metrics.functional import pearsonr
import torch.nn as nn


class PearsonCorrelation(_Loss):
    def __init__(self, ):
        super(PearsonCorrelation, self).__init__()

    def forward(self, z, data):
        z = z.reshape(z.shape[0], -1)
        data = data.reshape(data.shape[0], -1)
        corr = pearsonr(z, data).mean().abs()

        return corr


class LinearCorrelation(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(LinearCorrelation, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        # hardcoded to match the size of activation and input with respect to
        # split layer 6. TODO: Remove this hardcoding.
        channel_up_layer = nn.Conv2d(128, 192, 1).to(self.utils.device)
        self.client_model = nn.Sequential(self.client_model, channel_up_layer)
        self.channel_down_layer = nn.Conv2d(192, 128, 1).to(self.utils.device)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)
        self.down_optim = self.init_optim(config, self.channel_down_layer)
        self.loss = PearsonCorrelation()

        self.alpha = config["alpha"]
        self.cor_tag = "pearson_corr"
        #self.utils.logger.register_tag("train/" + self.cor_tag)
        #self.utils.logger.register_tag("val/" + self.cor_tag)

    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)
        self.x = x
        self._z = self.channel_down_layer(self.z)
        #z = self._z.detach()
        #z.requires_grad = True
        #self.cor_loss = self.loss(self.x, self.z)
        #self.utils.logger.add_entry(self.mode + "/" + self.cor_tag,
        #                            self.cor_loss.item())
        return self._z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        # Higher the alpha, higher the weight for cor loss would be
        self._z.backward((1 - self.alpha) * server_grads, retain_graph=True)
        (self.alpha * self.cor_loss).backward()
        self.optim.step()
