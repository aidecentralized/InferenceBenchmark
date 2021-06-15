import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from  torch.nn.modules.loss import _Loss
torch.autograd.set_detect_anomaly(True)
distance = nn.CrossEntropyLoss()

class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        if (input < 0).any() or (input > 1).any():
            print(input)
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(input * torch.log(input), dim=1))

        return H

def loss_fn(logits, labels):
    CE = distance(logits, labels)
    return CE


class ResNet18Client(nn.Module):
    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        self.split_layer = config["split_layer"]
        self.is_grid_crop = config["is_grid_crop"]

        self.model = models.resnet18(pretrained=False)

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)
        if self.is_grid_crop:
            self.im_size = 112
            self.window_size = 14

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.split_layer:
                break
            if i == 0 and self.is_grid_crop:
                batch_size = x.shape[0]
                out = []
                for n in range(batch_size):
                   for i in range(0, self.im_size, self.window_size):
                      for j in range(0, self.im_size, self.window_size):
                         patch = x[n, :, i:i+self.window_size, j:j+self.window_size]
                         patch = F.interpolate(patch.unsqueeze_(0), scale_factor=(8, 8), mode='bilinear', align_corners=False).squeeze_(0)
                         out.append(patch)
                out = torch.stack(out).float().to(x.device)
                out = l(out)
                x = []
                for i in range(0, out.shape[0], out.shape[1]):
                   x.append(out[i: i+out.shape[1]].sum(1).squeeze(1))
                x = torch.stack(x)
            else:
                #print(l, x.shape)
                x = l(x)
        # print(x.shape)
        return x


class ResNet18Server(nn.Module):
    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        return nn.functional.softmax(x, dim=1)


class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, config):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = config["pruning_ratio"]
        self.pruning_style = config["pruning_style"]
        # decoy layer to allow creation of optimizer
        self.decoy_layer = nn.Linear(10, 10)
        self.temp = 1/30

        if self.pruning_style == "network":
            self.logits = config["logits"]
            self.split_layer = config["split_layer"]
            self.model = models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)
        elif self.pruning_style == "noise":
            self.mean = -1.
            self.stddev = 2.

    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #print(exponent)
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        #print(answer)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        # pruning_vector = fmap_score.unsqueeze(dim=2).unsqueeze(dim=3)
        # x = x * pruning_vector
        try:
            index_array = torch.arange(num_channels).repeat(x.shape[0], 1).cuda() # commented out for cude?
        except:
            index_array = torch.arange(num_channels).repeat(x.shape[0], 1)
        indices = index_array[fmap_score < 0.5]
        return indices

    def noise_channels(self, x):
        noise = x.data.new(x.size()).normal_(self.mean, self.stddev)
        return x + noise

    def network_forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            #print(l, x.shape, x.device)
            x = l(x)
        #print(x.shape, x.device)
        return x

    def forward(self, x):
        if self.pruning_style == "random":
            indices = self.get_random_channels(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style in ["nopruning", "maxentropy", "adversarial"]:
            # decoy indices to prevent scheduler from breaking
            indices = torch.randint(1, 100, (x.shape[0], 30), device=x.device)
        elif self.pruning_style == "network":
            indices = self.get_channels_from_network(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style == "noise":
            indices = torch.randint(1, 100, (x.shape[0], 30), device=x.device)
            x = self.noise_channels(x)
        return x, indices
