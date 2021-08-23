from algos.deepobfuscator import DeepObfuscator
from utils.metrics import MetricLoader

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



class maxentropy(DeepObfuscator):
    """ The only difference between maxentropy and Deepobfuscator
    is the loss function for the proxy adversary and the label is the private attribute instead of reconstruction.
    """
    def __init__(self, config, utils) -> None:
        super(maxentropy, self).__init__(config, utils)
        self.update_loss()

    def update_loss(self):
        self.loss = EntropyLoss().forward

    def get_adv_loss(self):
        # Since it is L1, it has to be minimized
        return self.adv_loss