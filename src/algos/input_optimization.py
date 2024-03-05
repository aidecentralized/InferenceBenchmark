import json
from utils.config_utils import combine_configs, process_config
from algos.uniform_noise import UniformNoise
from algos.nopeek import NoPeek
from algos.simba_algo import SimbaAttack
from utils.metrics import MetricLoader
import torch
from torchvision.utils import save_image

class InputOptimization(SimbaAttack):
  def __init__(self, config, utils):
    super().__init__(utils)
    self.initialize(config)

  def initialize(self, config):
    self.attribute = config["attribute"]
    self.model_name = config["target_model"]

    self.img_size = config["img_size"]
    # load obfuscator model
    target_exp_config = json.load(open(config["target_model_config"])) #config_loader(config["model_config"])
    system_config = json.load(open("./configs/system_config.json")) #config_loader(config["model_config"])
    target_exp_config["client"]["challenge"] = True
    target_config = process_config(combine_configs(system_config, target_exp_config))
    self.target_config = target_config

    from interface import load_algo
    self.model = load_algo(target_config, self.utils)
    
    wts_path = config["target_model_path"]
    wts = torch.load(wts_path)
    if isinstance(self.model.client_model, torch.nn.DataParallel):  # type: ignore
        self.model.client_model.module.load_state_dict(wts)
    else:
        self.model.client_model.load_state_dict(wts)

    self.metric = MetricLoader(data_range=1)

    self.loss_tag = "recons_loss"

    self.ssim_tag = "ssim"
    self.utils.logger.register_tag("train/" + self.ssim_tag)

    self.l1_tag = "l1"
    self.utils.logger.register_tag("train/" + self.l1_tag)
    
    self.l2_tag = "l2"
    self.utils.logger.register_tag("train/" + self.l2_tag)

    self.psnr_tag = "psnr"
    self.utils.logger.register_tag("train/" + self.psnr_tag)


    self.iters = config["iters"]
    self.lr = config["lr"]

    if config["optimizer"] == "adam":
      self.optim = torch.optim.Adam  # type: ignore
    else:
      self.optim = torch.optim.SGD  # type: ignore

    if config["loss_fn"] == "ssim":
        self.loss_fn = self.metric.ssim
        self.sign = -1 # to maximize ssim
    elif config["loss_fn"] == "l1":
        self.loss_fn = self.metric.l1
        self.sign = 1 # to minimize l1
    elif config["loss_fn"] == "l2":
        self.loss_fn = self.metric.l2
        self.sign = 1 # to minimize l2
    elif config["loss_fn"] == "lpips":
        self.loss_fn = self.metric.lpips
        self.sign = 1 # to minimize lpips
    pass 

  def forward(self, items):
    if self.mode == "val":
      z = items["z"]
      img = items["img"]

      ys = torch.rand((z.shape[0], 3, 128, 128)).to(self.utils.device)  # type: ignore
      ys.requires_grad_(True)
      optim = self.optim([ys], lr=self.lr)
      
      prefix = "train/"
      self.utils.logger.set_log_freq(self.iters)
      
      # log the lowest loss
      lowest_loss = float("inf")
      lowest_ys = torch.clone(ys)  # type: ignore

      for _ in range(self.iters):
        optim.zero_grad()
        ys = torch.nn.functional.interpolate(ys, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        out = self.model({"x": ys})        
        loss = self.loss_fn(out, z)

        ssim = self.metric.ssim(img, ys)
        self.utils.logger.add_entry(prefix + self.ssim_tag,
                                      ssim.item())

        if ssim < lowest_loss: 
          lowest_loss = ssim
          lowest_ys = torch.clone(ys)  # type: ignore

        (self.sign * loss).backward()
        optim.step()

      return img
      ssim = self.metric.ssim(img, lowest_ys)
      self.utils.logger.add_entry(prefix + self.ssim_tag,
                                    ssim.item())
      self.utils.logger.flush_epoch()

  def backward(self, _):
    pass