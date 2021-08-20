from algos.deepobfuscator import DeepObfuscator
from utils.metrics import MetricLoader


class PAN(DeepObfuscator):
    """ The only difference between PAN and Deepobfuscator
    is the loss function for the proxy adversary
    """
    def __init__(self, config, utils) -> None:
        super(PAN, self).__init__(config, utils)
        self.update_loss()

    def update_loss(self):
        self.loss = MetricLoader().l1

    def get_adv_loss(self):
        # Since it is L1, it has to be minimized
        return self.adv_loss