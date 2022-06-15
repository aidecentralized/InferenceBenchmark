import torch
import torch.nn as nn
from algos.simba_algo import SimbaDefence
from models.image_decoder import Decoder
from utils.metrics import MetricLoader
from models.Xception import Xception


class AIOI(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(AIOI, self).__init__(utils)
        self.initialize(config)
        
    def get_secret_priors(self, config):
        if config["dataset"] == "fairface":
            if config["protected_attribute"] == "race":
                secret_prior = torch.tensor([0.1416, 0.1416, 0.1412, 0.1905, 0.1067, 0.1534, 0.1250]).to(self.utils.device)
                
            elif config["protected_attribute"] == "gender":
                secret_prior = torch.tensor([0.4797, 0.5203]).to(self.utils.device)
        return secret_prior
        

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)
        
        self.secret_prior = self.get_secret_priors(config)
        print("secret prior", self.secret_prior)

        self.proxy_adv_model = Xception(config["proxy_adversary"]["num_class"])
        self.utils.model_on_gpus(self.proxy_adv_model)
        self.utils.register_model("proxy_adv_model", self.proxy_adv_model)
        self.proxy_adv_optim = self.init_optim(config, self.proxy_adv_model)

        self.loss = MetricLoader().cross_entropy
        self.kl_loss = MetricLoader().KLdivergence
        self.lambda_ph = config["lambda_ph"]
        self.budget_ph = config["budget_ph"]
        self.adv_tag = "adv"
        self.utils.logger.register_tag("train/" + self.adv_tag)
        self.utils.logger.register_tag("val/" + self.adv_tag)



    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)

        adv_pred = self.proxy_adv_model(self.z)
        self.adv_loss = self.loss(adv_pred, items["prvt_lbls"])
        
        sec_prior = self.secret_prior
        
#         print("adv_prediction", adv_pred)
        print("sec_prior", sec_prior) 
        adv_pred = torch.sigmoid(adv_pred)
        
        
        self.penalty_secret_loss = torch.square(nn.ReLU()(self.kl_loss(adv_pred, sec_prior) - self.budget_ph))
        
        self.utils.logger.add_entry(self.mode + "/" + self.adv_tag,
                                    self.adv_loss.item())
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        self.proxy_adv_optim.zero_grad()

        # Backprop on adversary first
        adv_loss = self.adv_loss
        adv_loss.backward(retain_graph = True)
        self.proxy_adv_optim.step()

        penality_loss = self.lambda_ph*self.penalty_secret_loss
        penality_loss.backward(retain_graph = True)
        
        
        self.z.backward(server_grads)
        self.optim.step()

