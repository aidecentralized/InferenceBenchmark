from algos.simba_algo import SimbaDefence
from models.image_decoder import Decoder
from utils.metrics import MetricLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math

class NoisyActivation(nn.Module):
    def __init__(self,  given_locs, given_scales, min_scale, max_scale):
        super(NoisyActivation, self).__init__()
        size = given_scales.shape
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.given_locs = given_locs 
        self.given_scales = given_scales
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs).cuda())         
        self.rhos = nn.Parameter(torch.ones(size).cuda() - 5) #-inf
        self.normal = torch.distributions.normal.Normal(0,1)
        self.rhos.requires_grad = True
        self.locs.requires_grad = True
        
        
    def scales(self):
        return ((1.0 +torch.tanh(self.rhos))/2*(self.max_scale-self.min_scale) +self.min_scale).cuda()             
    
    def sample_noise(self):
        epsilon = self.normal.sample(self.rhos.shape).cuda()
        return  self.locs + self.scales()*epsilon
                                 
                                                   
    def forward(self, input):
        noise = self.sample_noise().cuda()
        return (input) + noise


class Cloak(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(Cloak, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        img_size = config["proxy_adversary"]["img_size"] # here there is no proxy adv, but this is used to get the image size -> change in the config file
        mus = torch.zeros((3,img_size,img_size)) # image size
        scale = torch.ones((3,img_size,img_size))*0.001
        
        min_scale = config["min_scale"]
        max_scale = config["max_scale"]
        self.coeff = config["coeff"]
        
        """ model_features: this is the feature extraction part of the VGG network as discussed in the CLOAK paper -> we can use any other model like the resnet etc as well
        model_classifier: this is the fully_connected layer or the FC sub-section of the VGG16 (modified) network
        min_scale: default value is 0
        max_scale: default value is 5
        given_locs: default is the "mus" variable intialised in cloak class
        given_scale: default is the "scale" variable intialised in cloak class
        All this information must be added to the config file for verifying this
        model_syn = vgg_syn(model.convnet, model.fc ,0, 5 ,mus, scale ) -> the way the model is init"""        

        self.client_model = NoisyActivation(mus, scale, min_scale, max_scale)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)

#         config["img_size"] = img_size

        
#         self.loss = MetricLoader().ssim
#         self.adv_tag = "adv"
#         self.utils.logger.register_tag("train/" + self.adv_tag)
#         self.utils.logger.register_tag("val/" + self.adv_tag)
        

# In case you are not running on a parrallel infra remove the "module" from the below 2 lines -> this is used to access objects when it is in DataParallel form
                
        self.client_model.module.rhos.requires_grad = True
        self.client_model.module.locs.requires_grad = True



    def forward(self, items):
        self.z = self.client_model(items["x"])
        z = self.z
        if self.detached:
            z = z.detach()
            z.requires_grad = True
        return z 

    def backward(self, items):
        server_grads = items["server_grads"]
        noise_loss = (-1)*self.coeff*torch.log(torch.mean((self.client_model.module.scales()))) + 10
        noise_loss.backward(retain_graph = True)

        self.optim.zero_grad()
        self.z.backward(server_grads)
        self.optim.step()





