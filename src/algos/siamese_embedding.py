import torch
from algos.simba_algo import SimbaDefence
from torch.nn.modules.loss import _Loss


class ContrastiveLoss(_Loss):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_mask(self, labels):
        labels = labels.unsqueeze(-1).to(dtype=torch.float64)
        class_diff = torch.cdist(labels, labels, p=1.0)
        return torch.clamp(class_diff, 0, 1)

    def get_pairwise(self, z):
        z = z.view(z.shape[0], -1)
        return torch.cdist(z, z, p=2.0)

    def forward(self, z, labels):
        mask = self.get_mask(labels).to(z.device)
        pairwise_dist = self.get_pairwise(z)
        loss = (1 - mask) * pairwise_dist +\
               mask * torch.maximum(torch.tensor(0.).to(z.device), self.margin - pairwise_dist)
        return loss.mean()


class SiameseEmbedding(SimbaDefence):
    """ Introduced in the paper https://arxiv.org/pdf/1703.02952.pdf and
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8962332
    """
    def __init__(self, config, utils) -> None:
        super(SiameseEmbedding, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)

        self.loss = ContrastiveLoss(config["margin"])
        self.alpha = config["alpha"]
        self.ct_loss_tag = "ct_loss"
        self.utils.logger.register_tag("train/" + self.ct_loss_tag)
        self.utils.logger.register_tag("val/" + self.ct_loss_tag)

    def forward(self, items):
        x = items["x"]
        pred_lbls = items["pred_lbls"]
        self.z = self.client_model(x)
        self.contrastive_loss = self.loss(self.z, pred_lbls)
        self.utils.logger.add_entry(self.mode + "/" + self.ct_loss_tag,
                                    self.contrastive_loss.item())
        # z will be detached to prevent any grad flow from the client
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        (self.alpha * self.contrastive_loss).backward(retain_graph=True)
        self.z.backward((1 - self.alpha) * server_grads)
        self.optim.step()
