import torch
from torch import nn
import pytorch_msssim
import lpips

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


class MetricLoader():
    # Add more metrics
    def __init__(self, device=None):
        self.l1_dist = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self._ssim = pytorch_msssim.SSIM()
        self.l2_dist = nn.MSELoss()
        self._psnr = PSNR()
        _lpips = lpips.LPIPS(net='vgg')
        if device is None:
            self._lpips = _lpips.cuda()
        else:
            self._lpips = _lpips.to(device)

    def acc(self, preds, y):
        return (preds.argmax(dim=1) == y).sum().item() / preds.shape[0]

    def l1(self, img1, img2):
        return self.l1_dist(img1, img2)

    def cross_entropy(self, preds, lbls):
        return self.ce(preds, lbls)

    def ssim(self, img1, img2):
        return self._ssim(img1, img2)

    def l2(self, img1, img2):
        return self.l2_dist(img1, img2)

    def psnr(self, img1, img2):
        return self._psnr(img1, img2)

    def lpips(self, img1, img2):
        score = self._lpips(img1, img2)
        return score.mean()