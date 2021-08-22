from algos.deepobfuscator import DeepObfuscator
from utils.metrics import MetricLoader


class maxentropy(DeepObfuscator):
    """ The only difference between maxentropy and Deepobfuscator
    is the loss function for the proxy adversary and the label is the private attribute instead of reconstruction.
    """
    def __init__(self, config, utils) -> None:
        super(maxentropy, self).__init__(config, utils)
        self.update_loss()

    def update_loss(self):
        self.loss = MetricLoader().l1

    def get_adv_loss(self):
        # Since it is L1, it has to be minimized
        return self.adv_loss