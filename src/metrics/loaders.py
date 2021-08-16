

class MetricLoader():
    def __init__(self, config) -> None:
        self.epoch = 0
        self.iters = 0

    def acc(self, preds, y):
        return (preds.argmax(dim=1) == y).sum().item()
