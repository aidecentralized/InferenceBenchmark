import json
from utils.config_utils import process_config
from algos.uniform_noise import UniformNoise
from algos.nopeek import NoPeek
from algos.simba_algo import SimbaAttack
from utils.metrics import MetricLoader
import torch
from torchvision.utils import save_image
from models.skip import skip
import os




class InputModelOptimization(SimbaAttack):
  def __init__(self, config, utils):
    super().__init__(utils)
    self.initialize(config)

  def create_gen_model(self):
    gen_model = skip(3, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                                      num_channels_up =   [16, 32, 64, 128, 128, 128],
                                      num_channels_skip = [4, 4, 4, 4, 4, 4],
                                      filter_size_down = [7, 7, 5, 5, 3, 3],  # type: ignore
                                      filter_size_up = [7, 7, 5, 5, 3, 3],   # type: ignore
                      upsample_mode='nearest', downsample_mode='avg', need_sigmoid=True, pad='zero', act_fun='LeakyReLU') #.type(torch.cuda.FloatTensor)
    gen_model.to(self.utils.device)
    return gen_model


  def initialize(self, config):
    self.attribute = config["attribute"]
    self.obf_model_name = config["target_model"]

    # load obfuscator model
    target_exp_config = json.load(open(config["target_model_config"])) #config_loader(config["model_config"])
    system_config = json.load(open("./configs/system_config.json")) #config_loader(config["model_config"])
    target_config = process_config(system_config, target_exp_config)
    self.target_config = target_config

    from interface import load_algo
    self.obf_model = load_algo(target_config, self.utils)

    wts_path = self.target_config["model_path"] + "/client_model.pt"
    wts = torch.load(wts_path)
    if isinstance(self.obf_model.client_model, torch.nn.DataParallel):  # type: ignore
        self.obf_model.client_model.module.load_state_dict(wts)
    else:
        self.obf_model.client_model.load_state_dict(wts)
    
    self.obf_model.enable_logs(False)
    self.obf_model.set_detached(False)

    self.model = self.obf_model # to prevent errors thrown

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

    self.save_images = True
    self.save_image_path = os.path.join(target_config["log_path"], "images")
    print(self.save_image_path)

  def save_image(self, image, name):
    dir = self.save_image_path
    if not os.path.exists(dir):
      os.makedirs(dir)
    path = os.path.join(dir, name)
    save_image(image, path)

  def forward(self, items):
    if self.mode == "val":
      z = items["z"]
      img = items["img"]

      gen_model = self.create_gen_model()
      rand_inp_og = torch.rand((z.shape[0], 3, 128, 128)).detach().to(self.utils.device)
      inp_noise = rand_inp_og.detach().clone()

      optim = self.optim(gen_model.parameters(), lr=self.lr)
      
      prefix = "train/"
      self.utils.logger.set_log_freq(self.iters)
      
      # log the lowest loss
      best_ssim = 0
      best_ys = gen_model(rand_inp_og)[:,:,:128, :128]


      
      for i in range(self.iters):
        # add noise in input space
        if i < 10000:
            rand_inp = rand_inp_og + (inp_noise.normal_() * 10)
            for n in [x for x in gen_model.parameters() if len(x) == 4]:
                n = n + n.detach().clone().normal_()*n.std()/50
        elif i < 15000:
            rand_inp = rand_inp_og + (inp_noise.normal_() * 2)
            for n in [x for x in gen_model.parameters() if len(x) == 4]:
                n = n + n.detach().clone().normal_()*n.std()/50
        elif i < 20000:
            rand_inp = rand_inp_og + (inp_noise.normal_() / 2)
            for n in [x for x in gen_model.parameters() if len(x) == 4]:
                n = n + n.detach().clone().normal_()*n.std()/50
        else:
          rand_inp = rand_inp_og
  
        rand_inp = rand_inp.to(self.utils.device)

        optim.zero_grad()
        ys = gen_model(rand_inp)[:,:,:128, :128]
        out = self.obf_model({"x": ys})
        loss = self.loss_fn(out, z)

        ssim = self.metric.ssim(img, ys)
        self.utils.logger.add_entry(prefix + self.l2_tag,
                                      ssim.item())

        if ssim > best_ssim: 
          best_ssim = ssim
          best_ys = torch.clone(ys) 

        # if self.utils.logger.curr_iters % self.utils.logger.trigger_freq == 0:
        #   print("SSIM: ", ssim.item())
        
        if self.save_images and self.utils.logger.curr_iters % (self.utils.logger.trigger_freq) == 0:
          self.save_image(ys, f"{self.utils.logger.epoch}_ys_{self.utils.logger.curr_iters}.png")

        (self.sign * loss).backward()
        optim.step()
      
      if self.save_images:
        self.save_image(img, f"{self.utils.logger.epoch}_img.png")
        self.save_image(best_ys, f"{self.utils.logger.epoch}_ys.png")
      
      self.utils.logger.flush_epoch()

      

  def backward(self, _):
    pass
