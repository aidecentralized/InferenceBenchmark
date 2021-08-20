import torch
from algos.simba_algo import SimbaDefence
from models.image_decoder import Decoder
from utils.metrics import MetricLoader


class DeepObfuscator(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(DeepObfuscator, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        img_size = config["proxy_adversary"]["img_size"]
        self.client_model = self.init_client_model(config)
        channels, patch_size = self.generate_patch_params(self.client_model, img_size)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)

        config["img_size"] = img_size
        config["channels"] = channels
        config["patch_size"] = patch_size
        self.proxy_adv_model = Decoder(config)
        self.utils.model_on_gpus(self.proxy_adv_model)
        self.utils.register_model("proxy_adv_model", self.proxy_adv_model)
        self.proxy_adv_optim = self.init_optim(config, self.proxy_adv_model)

        self.loss = MetricLoader().ssim
        self.alpha = config["alpha"]
        self.adv_tag = "adv"
        self.utils.logger.register_tag("train/" + self.adv_tag)
        self.utils.logger.register_tag("val/" + self.adv_tag)

    def generate_patch_params(self, model, img_size):
        # 3 because of RGB assumption. 1 is batch size
        img = torch.randn(1, 3, img_size, img_size)
        patch = model(img)
        assert patch.shape[2] == patch.shape[3] # only square images
        return patch.shape[1], patch.shape[2]

    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)

        x_recons = self.proxy_adv_model(self.z)
        self.adv_loss = self.loss(x_recons, x)
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
        adv_loss = self.get_adv_loss()
        adv_loss.backward(retain_graph = True)
        self.proxy_adv_optim.step()

        # Higher the alpha, higher the weight for adv loss would be
        # Flip the gradient sign for maximizing proxy adversary's loss
        for params in self.client_model.parameters():
            params.grad *= -1 * self.alpha
        self.z.backward((1 - self.alpha) * server_grads)
        self.optim.step()

    def get_adv_loss(self):
        # Since it is ssim, it has to be maximized
        return -1 * self.adv_loss
