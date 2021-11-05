from algos.simba_algo import SimbaDefence
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from models.complex_models import Discriminator, RealToComplex, ComplexToReal, ResNetEncoderComplex, ResNetDecoderComplex

def get_encoder_output_size(encoder, dims):
    x = torch.randn((1,)+dims)
    with torch.no_grad():
        out = encoder(x)
    if type(out) == tuple:
        out = out[0]
    return list(out.size())[1:]

class ComplexNN(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(ComplexNN, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.optimizer_idx = 0
        self.encoder_model,self.decoder_model = self.init_client_model(config)
        img_size = config["img_size"]
        size = get_encoder_output_size(self.encoder_model, (3,img_size,img_size))
        self.discriminator = Discriminator(size=size)
        models = [self.encoder_model, self.decoder_model, self.discriminator]
        self.put_on_gpus(models)

        self.utils.register_model("encoder_model", self.encoder_model)
        self.utils.register_model("discriminator_model", self.discriminator)
        self.utils.register_model("decoder_model", self.decoder_model)
        self.optim_encoder , self.optim_decoder , self.optim_discriminator = self.init_optim(config, self.encoder_model, self.decoder_model, self.discriminator)
        
        self.real_to_complex = RealToComplex()
        self.complex_to_real = ComplexToReal()
        self.loss_fn = F.cross_entropy
        self.alpha = config["alpha"]
        self.k = config["k"]

        self.loss_tag = "decoder_loss"
        self.acc_tag = "decoder_acc"
        tags = [self.loss_tag, self.acc_tag]
        for tag in tags:
            self.utils.logger.register_tag("train/" + tag)
            self.utils.logger.register_tag("val/" + tag)
    
    def put_on_gpus(self,models):
        for model in models:
            model = self.utils.model_on_gpus(model)

    def init_client_model(self, config):
        if config["model_name"] == "resnet20complex":
            encoder_model = ResNetEncoderComplex(3)
            decoder_model = ResNetDecoderComplex(3, config["logits"], "alpha") 
        else:
            print("can't find complex client model")
            exit()

        return encoder_model,decoder_model

    def init_optim(self, config, encoder, decoder, discriminator):
        encoder_parameters = encoder.parameters()
        decoder_parameters = decoder.parameters()

        if config["optimizer"] == "adam":
            optimizer_e = torch.optim.Adam(encoder_parameters, 
                lr=config["lr"],
            )
            
            optimizer_decoder = torch.optim.Adam(decoder_parameters)
            
            optimizer_discriminator = torch.optim.Adam(
                                            discriminator.parameters(),
                                            lr=config["lr"],
                                            )
        else:
            print("Unknown optimizer {}".format(config["optimizer"]))
        return optimizer_e,optimizer_decoder,optimizer_discriminator
    
    def train(self):
        self.mode = "train"
        self.encoder_model.train()
        self.decoder_model.train()

    def eval(self):
        self.mode = "val"
        self.encoder_model.eval()
        self.decoder_model.eval()

    def forward(self, items):
        inp = items["x"]
        # Pass through encoder
        a = self.encoder_model(inp)
        self.a = a  
        # Shuffle batch elements of a to create b
        with torch.no_grad():
            indices = np.random.permutation(a.size(0))
            b = a[indices]
        
        self.z, self.theta = self.real_to_complex(a,b) 
        
        # Get discriminator score expectation over k rotations
        self.score_fake = 0
        for k in range(self.k):
            # Shuffle batch to get b
            indices = np.random.permutation(a.size(0))
            b = a[indices]

            # Rotate a
            x, _ = self.real_to_complex(a,b)
            a_rotated = x[:,0]
            # Get discriminator score  
            self.score_fake += self.discriminator(a_rotated)

        self.score_fake /= self.k # Average score
        z = self.z.detach()
        z.requires_grad = True
        return z
    
    def infer(self, h, labels):
        h.retain_grad()
        y = self.complex_to_real(h,self.theta)
        y.retain_grad()
        self.preds = self.decoder_model(y)
        self.acc = (self.preds.argmax(dim=1) == labels).sum().item() / self.preds.shape[0]
        self.utils.logger.add_entry(self.mode + "/" + self.acc_tag, self.acc)
        if self.optimizer_idx%2 == 0:
            g_loss_adv = -torch.mean(self.score_fake)
            g_loss_ce = self.loss_fn(self.preds,labels)
            loss = g_loss_adv + g_loss_ce
            self.optim_decoder.zero_grad()
            loss.backward(retain_graph=True)
            self.optim_decoder.step()
            self.utils.logger.add_entry(self.mode + "/" + self.loss_tag, loss.item())
            return h.grad
        else:    
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            self.d_loss_adv = -torch.mean(self.discriminator(self.a)) + torch.mean(self.score_fake)
            self.optim_discriminator.zero_grad()
            self.d_loss_adv.backward()
            self.optim_discriminator.step()
            return None

    def backward(self, items):
        if self.optimizer_idx%2 == 0:
            self.optim_encoder.zero_grad()
            self.z.backward(items["server_grads"])
            self.optim_encoder.step()
        self.optimizer_idx += 1


