from interface import (load_config, load_algo, load_data, load_model, load_utils)


class Scheduler():
    def __init__(self, config_path) -> None:
        self.initialize(config_path)

    def initialize(self, config_path) -> None:
        self.config = load_config(config_path)
        self.utils = load_utils(self.config)
        self.algo = load_algo(self.config, self.utils)
        self.dataloader = load_data(self.config)
        self.model = load_model(self.config, self.utils)

    def run_job(self):
        self.utils.copy_source_code(self.config["results_path"])
        for self.metric.epoch in range(self.config["total_epochs"]):
            self.train()
            self.test()
            self.epoch_summary()
        self.generate_challenge()

    def train(self) -> None:
        for batch_idx, sample in enumerate(self.dataloader.train):
            data, pred_lbls, privt_lbls = self.utils.get_data(sample)
            z = self.algo.forward(data)
            grads = self.model.processing(z, pred_lbls)
            self.algo.backward(grads, privt_lbls)
            #self.metric.log_data()

    def test(self) -> None:
        for batch_idx, sample in enumerate(self.dataloader.test):
            data, pred_lbls, _ = self.utils.get_data(sample)
            z = self.algo.forward(data)
            self.model.processing(z, pred_lbls)
            self.metric.log_data()

    def epoch_summary():
        pass

    def generate_challenge(self) -> None:
        for batch_idx, sample in enumerate(self.dataloader.test):
            data, pred_lbls, privt_lbls = self.utils.get_data(sample)
            z = self.algo.forward(data)
            self.utils.save_data(z, pred_lbls)
