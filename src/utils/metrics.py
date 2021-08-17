class MetricLoader():
    # Add more metrics
    def __init__(self):
        pass

    def acc(self, preds, y):
        return (preds.argmax(dim=1) == y).sum().item() / preds.shape[0]
