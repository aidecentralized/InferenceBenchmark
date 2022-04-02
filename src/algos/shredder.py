from algos.simba_algo import SimbaDefence
from models.image_decoder import Decoder
from utils.metrics import MetricLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math


class NoisyActivation(nn.Module):
    def __init__(self, activation_size):
        super(NoisyActivation, self).__init__()

        m =torch.distributions.laplace.Laplace(loc = 0.6, scale = 1.2, validate_args=None)
        self.noise = nn.Parameter(m.rsample(activation_size))

    def forward(self, input):  
        device = input.device
        return input + self.noise.cuda(device)


class Shredder(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(Shredder, self).__init__(utils)
        self.initialize(config)
        
    def activation_shape(self, model, img_size):
        img = torch.randn(1, 3, img_size, img_size)
        patch = model(img)
        assert patch.shape[2] == patch.shape[3]
        
        return patch.shape[1:]
    

    def initialize(self, config):
        img_size = config["proxy_adversary"]["img_size"]
    
        self.client_model = self.init_client_model(config)
        for params in self.client_model.parameters():
            params.requires_grad = False

        activation_size = self.activation_shape(self.client_model, img_size)
        self.shredder_noise = NoisyActivation(activation_size)
    
#         self.client_model = nn.Sequential(*nn.ModuleList(list(self.client_model.children())[:config["split_layer"]]), self.shredder_noise)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)

        self.optim = self.init_optim(config, self.client_model)
        self.coeff = config["coeff"]
        
        

        config["img_size"] = img_size


    def forward(self, items):
        x_ = self.client_model(items["x"])
        self.z = self.shredder_noise(x_)
        
        
        z = self.z.detach()
        z.requires_grad = True
        
        return z 

    def backward(self, items):
        server_grads = items["server_grads"]
#         noise_loss = (-1)*self.coeff*(1/(torch.std(self.client_model.module.shredder_noise.noise)))
        noise_loss = (-1)*self.coeff*(1/(torch.std(self.shredder_noise.noise)))

        noise_loss.backward(retain_graph = True)

        self.optim.zero_grad()
        self.z.backward(server_grads)
        self.optim.step()






