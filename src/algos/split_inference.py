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

    def forward(self, items):
        x = items["x"]
        self.z = self.client_model(x)
        # z will be detached to prevent any grad flow from the client
        z = self.z.detach()
        z.requires_grad = True
        return z

    def backward(self, items):
        server_grads = items["server_grads"]
        self.optim.zero_grad()
        self.z.backward(server_grads)
        self.optim.step()
