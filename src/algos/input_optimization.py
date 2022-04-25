import json
from utils.config_utils import process_config
from algos.uniform_noise import UniformNoise
from algos.nopeek import NoPeek
from algos.simba_algo import SimbaAttack
from utils.metrics import MetricLoader
from utils.utils import Utils
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

class InputOptimization(SimbaAttack):
  def __init__(self, config, utils):
    super().__init__(utils)
    self.initialize(config)

  def initialize(self, config):
    self.attribute = config["attribute"]
    self.model_name = config["model"]

    # load obfuscator model
    target_config_file = open(config["model_config"])
    target_exp_config = json.load(target_config_file) #config_loader(config["model_config"])
    system_config_file = open("./configs/system_config.json")
    system_config = json.load(system_config_file) #config_loader(config["model_config"])
    target_config = process_config(system_config, target_exp_config)
    self.target_config = target_config
    print(target_config)
    # target_utils = load_utils()

    if self.model_name == "uniform_noise":
      self.model = UniformNoise(target_config["client"], self.utils)
    elif self.model_name == "nopeek": 
      self.model = NoPeek(target_config["client"], self.utils)
    else:
      raise NotImplementedError
    
    wts_path = self.target_config["model_path"] + "/client_model.pt"
    wts = torch.load(wts_path)
    if isinstance(self.model.client_model, torch.nn.DataParallel):
        self.model.client_model.module.load_state_dict(wts)
    else:
        self.model.client_model.load_state_dict(wts)

    # print(self.model)
    # print(self.utils.model_registry)
    # print(target_config["model_path"])




    # wts_path = target_config["model_path"] + "/client_model.pt"
    # # self.utils.load_saved_model(, self.model.client_model)

    # wts = torch.load(wts_path)
    # if isinstance(self.model.client_model, torch.nn.DataParallel):
    #     self.model.client_model.module.load_state_dict(wts)
    # else:
    #     self.model.client_model.load_state_dict(wts)





    # self.utils.load_saved_models()
    # self.model = self.utils.model_registry["client_model"]

    # print(self.model)
    # if self.model_name not in self.utils.model_registry:
    #   self.utils.load_saved_models()

    # assert self.model_name in self.utils.model_registry, f"MODEL {self.model_name} not found!"

    # self.model = self.utils.model_registry[self.model_name]


    self.metric = MetricLoader()

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
      self.optim = torch.optim.Adam
    else:
      self.optim = torch.optim.SGD

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

  def getBack(self, var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:') #, tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                self.getBack(n[0])
    
  def forward(self, items):

    """
    PROBLEMS:
    - ssim for img and untrained ys is already 0.9888
    - loss(z, out) decreases, but does not improve loss between img and ys
    
    """
    if self.mode == "val":
      # print("TESTING")
      z = items["z"]
      img = items["img"]
      # grab batch size + image size to get these dims
      ys = torch.rand((z.shape[0], 3, 128, 128)).to(self.utils.device)
      
      # ys = torch.rand((128, 128, 16, 16)).to(self.utils.device)
      print(z.shape, img.shape, ys.shape)
      ys.requires_grad_(True)
      optim = self.optim([ys], lr=self.lr)
      
      # model_temp = torch.nn.Linear(16,16).to(self.utils.device)
      prefix = "train/"
      self.utils.logger.set_log_freq(self.iters)
      # log the lowest loss
      # last = torch.clone(ys)

      ssim = self.metric.ssim(img, ys)
      l2 = self.metric.l2(img, ys)
      
      print("SSIM LOSS: ", ssim.item())
      print("L2 LOSS: ", l2.item())

      save_image(ys[0], 'ys_0.png')
      save_image(img[0], 'img_0.png')


      for _ in range(self.iters):
        optim.zero_grad()
        out = self.model({"x": ys})
        # out = model_temp(ys)
        loss = self.loss_fn(out, z)
        # print(loss)
        # self.getBack(loss.grad_fn)
        # print(out.shape, z.shape)
        # print("DIFF: ", self.loss_fn(ys, last))
        # last = torch.clone(ys)

        # ssim = self.metric.ssim(img, ys)
        # self.utils.logger.add_entry(prefix + self.ssim_tag,
        #                             ssim.item())

        # l1 = self.metric.l1(img, ys)
        # self.utils.logger.add_entry(prefix + self.l1_tag,
        #                             l1.item())

        l2 = self.metric.l2(img, ys)
        self.utils.logger.add_entry(prefix + self.l2_tag,
                                    l2.item())

        # psnr = self.metric.psnr(img, ys)
        # self.utils.logger.add_entry(prefix + self.psnr_tag,
        #                             psnr.item())
        (self.sign * loss).backward()

        optim.step()
      ssim = self.metric.ssim(img, ys)
      self.utils.logger.add_entry(prefix + self.ssim_tag,
                                    ssim.item())

      self.utils.logger.flush_epoch()
      save_image(ys[0], 'ys_0_after.png')


      

  def backward(self, _):
    pass