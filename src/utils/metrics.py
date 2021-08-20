from torch import nn
import pytorch_msssim

class MetricLoader():
    # Add more metrics
    def __init__(self):
        pass

    def acc(self, preds, y):
        return (preds.argmax(dim=1) == y).sum().item() / preds.shape[0]

    def l1(self, img1, img2):
        return nn.L1Loss()(img1, img2)

    def cross_entropy(self, preds, lbls):
        return nn.CrossEntropyLoss()(preds, lbls)

    def ssim(self, img1, img2):
        return pytorch_msssim.SSIM()(img1, img2)