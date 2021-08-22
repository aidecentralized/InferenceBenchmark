from algos.simba_algo import SimbaAttack
from models.image_decoder import Decoder
from utils.metrics import MetricLoader


class SupervisedDecoder(SimbaAttack):
    def __init__(self, config, utils):
        super(SupervisedDecoder, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        if config["attribute"] == "data":
            self.loss_tag = "recons_loss"
        else:
            self.loss_tag = "attribute_loss"
        self.utils.logger.register_tag("train/" + self.loss_tag)
        self.utils.logger.register_tag("val/" + self.loss_tag)

        self.model = Decoder(config)
        self.utils.model_on_gpus(self.model)
        self.utils.register_model("adv_model", self.model)
        self.optim = self.init_optim(config, self.model)

        if config["loss_fn"] == "ssim":
            self.loss_fn = MetricLoader().ssim
            self.sign = -1 # to maximize ssim
        elif config["loss_fn"] == "l1":
            self.loss_fn = MetricLoader().l1
            self.sign = 1 # to minimize l1

    def forward(self, items):
        z = items["z"]
        self.x = self.model(z)

    def backward(self, items):
        x = items["x"]
        loss = self.loss_fn(self.x, x)

        self.optim.zero_grad()
        (self.sign * loss).backward()
        self.optim.step()

        self.utils.logger.add_entry(self.mode + "/" + self.loss_tag,
                                    loss.item())