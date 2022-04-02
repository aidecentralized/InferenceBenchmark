from algos.simba_algo import SimbaAttack
from models.image_decoder import Decoder
from utils.metrics import MetricLoader


class SupervisedDecoder(SimbaAttack):
    def __init__(self, config, utils):
        super(SupervisedDecoder, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
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

        self.model = Decoder(config)
        self.utils.model_on_gpus(self.model)
        self.utils.register_model("adv_model", self.model)
        self.optim = self.init_optim(config, self.model)

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
        z = items["z"]
        self.x = self.model(z)
        x = items["x"]

        self.loss = self.loss_fn(self.x, x)
        self.utils.logger.add_entry(self.mode + "/" + self.loss_tag,
                                    self.loss.item())
    
        return self.x
    
    def backward(self, items):
        if self.mode == "val" and self.attribute == "data":
            prefix = "val/"

            ssim = self.metric.ssim(self.x, x)
            self.utils.logger.add_entry(prefix + self.ssim_tag,
                                        ssim.item())

            l1 = self.metric.l1(self.x, x)
            self.utils.logger.add_entry(prefix + self.l1_tag,
                                        l1.item())

            l2 = self.metric.l2(self.x, x)
            self.utils.logger.add_entry(prefix + self.l2_tag,
                                        l2.item())

            psnr = self.metric.psnr(self.x, x)
            self.utils.logger.add_entry(prefix + self.psnr_tag,
                                        psnr.item())

    def backward(self, _):
        self.optim.zero_grad()
        (self.sign * self.loss).backward()
        self.optim.step()
