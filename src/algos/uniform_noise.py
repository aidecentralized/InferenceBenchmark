from algos.simba_algo import SimbaDefence
import torch
from torch.distributions.laplace import Laplace

class AddGaussianNoise(object):
    def __init__(self, mean, sigma):
        self.sigma = sigma
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.sigma + self.mean


class AddLaplaceNoise(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, tensor):
        mean = torch.zeros(tensor.size()) + self.mean
        sigma = torch.zeros(tensor.size()) + self.sigma
        return tensor + Laplace(mean, sigma).sample().to(tensor.device)


class UniformNoise(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(UniformNoise, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)
        self.set_noise_params(config)

    def set_noise_params(self, config):
        if config["distribution"] == "gaussian":
            self.NoiseModel = AddGaussianNoise(mean=config["mean"], sigma=config["sigma"])
        elif config["distribution"] == "laplace":
            self.NoiseModel = AddLaplaceNoise(mean=config["mean"], sigma=config["sigma"])

    def forward(self, items):
        x = items["x"]
        _z = self.client_model(x)
        self.z = self.NoiseModel(_z)
        # z will be detached to prevent any grad flow from the client
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        self.z.backward(server_grads)
        self.optim.step()
