import torch
from algos.simba_algo import SimbaAttack
from models.image_decoder import Decoder
from models.model_zoo import get_resnet18
from utils.metrics import MetricLoader


class DiscriminatorAttack(SimbaAttack):
    def __init__(self, config, utils):
        super(DiscriminatorAttack, self).__init__(utils)
        self.initialize(config)

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        #print real_data.size()
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = self.utils.tensor_on_gpu(alpha)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = self.utils.tensor_on_gpu(interpolates)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def initialize(self, config):
        self.critic_iter = 0
        self.attribute = config["attribute"]
        self.metric = MetricLoader()
        if self.attribute == "data":
            self.loss_tag = "recons_loss"

            self.ssim_tag = "ssim"
            self.utils.logger.register_tag("val/" + self.ssim_tag)

            self.l1_tag = "l1"
            self.utils.logger.register_tag("val/" + self.l1_tag)
            
            self.l2_tag = "l2"
            self.utils.logger.register_tag("val/" + self.l2_tag)

            self.psnr_tag = "psnr"
            self.utils.logger.register_tag("val/" + self.psnr_tag)
        else:
            self.loss_tag = "attribute_loss"
        self.utils.logger.register_tag("train/" + self.loss_tag)
        self.utils.logger.register_tag("val/" + self.loss_tag)

        # Decoder model
        self.decoder = Decoder(config)
        self.utils.model_on_gpus(self.decoder)
        self.utils.register_model("adv_model", self.decoder)
        self.dec_optim = self.init_optim(config, self.decoder)
        self.model = self.decoder # additional pointer

        # Discriminator model
        self.discriminator = get_resnet18(1)
        self.utils.model_on_gpus(self.discriminator)
        self.utils.register_model("discriminator", self.discriminator)
        self.disc_optim = self.init_optim(config, self.discriminator)

        if config["loss_fn"] == "ssim":
            self.loss_fn = self.metric.ssim
            self.sign = -1 # to maximize ssim
        elif config["loss_fn"] == "l1":
            self.loss_fn = self.metric.l1
            self.sign = 1 # to minimize l1
        elif config["loss_fn"] == "lpips":
            self.loss_fn = self.metric.lpips
            self.sign = 1 # to minimize lpips

    def forward(self, items):
        z = self.utils.tensor_on_gpu(torch.rand(items["z"].size()))
        self.reconstruction = self.decoder(z)
        i = torch.randperm(z.shape[0])
        self.orig = items["x"]
        permuted_orig = self.orig[i]

        self.critic_iter += 1

        if not self.critic_iter % 5 == 0:
            self.d_real = self.discriminator(permuted_orig)
        self.d_fake = self.discriminator(self.reconstruction)
        
        self.loss = self.loss_fn(self.reconstruction, self.orig)

        if self.mode == "val" and self.attribute == "data":
            prefix = "val/"

            ssim = self.metric.ssim(self.reconstruction, self.orig)
            self.utils.logger.add_entry(prefix + self.ssim_tag,
                                        ssim.item())

            l1 = self.metric.l1(self.reconstruction, self.orig)
            self.utils.logger.add_entry(prefix + self.l1_tag,
                                        l1.item())

            l2 = self.metric.l2(self.reconstruction, self.orig)
            self.utils.logger.add_entry(prefix + self.l2_tag,
                                        l2.item())

            psnr = self.metric.psnr(self.reconstruction, self.orig)
            self.utils.logger.add_entry(prefix + self.psnr_tag,
                                        psnr.item())

        self.utils.logger.add_entry(self.mode + "/" + self.loss_tag,
                                    self.loss.item())
        return self.reconstruction

    def backward(self, _):
        # Generator training
        self.disc_optim.zero_grad()
        self.dec_optim.zero_grad()
        if self.critic_iter % 5 == 0:
            (-1*self.d_fake).mean().backward()
            self.dec_optim.step()
        else:
            self.d_fake.mean().backward(retain_graph=True)
            (-1*self.d_real).mean().backward(retain_graph=True)
            self.calc_gradient_penalty(self.discriminator, self.orig, self.reconstruction).backward()
            self.disc_optim.step()        
