from algos.simba_algo import SimbaDefence


class SplitInference(SimbaDefence):
    def __init__(self, config, utils) -> None:
        super(SplitInference, self).__init__(utils)
        self.initialize(config)

    def initialize(self, config):
        self.client_model = self.init_client_model(config)
        self.put_on_gpus()
        self.utils.register_model("client_model", self.client_model)
        self.optim = self.init_optim(config, self.client_model)

    def train(self):
        self.mode = "train"
        self.client_model.train()

    def eval(self):
        self.mode = "val"
        self.client_model.eval()

    def forward(self, x):
        self.z = self.client_model(x)
        # z will be detached to prevent any grad flow from the client
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, server_grads, privt_lbls):
        self.optim.zero_grad()
        self.z.backward(server_grads)
        self.optim.step()

    def log_metrics(self):
        pass
