from interface import (load_config, load_algo, load_data, load_model, load_utils)
import torch

class Scheduler():
    def __init__(self, config_path) -> None:
        self.initialize(config_path)

    def initialize(self, config_path) -> None:
        self.config = load_config(config_path)
        self.utils = load_utils(self.config)

        if not self.config["experiment_type"] == "challenge":
            self.utils.copy_source_code(self.config["results_path"])
        self.utils.init_logger()

        self.dataloader = load_data(self.config)

        self.algo = load_algo(self.config, self.utils, self.dataloader.train)

        if self.config["experiment_type"] == "defense":
            self.model = load_model(self.config, self.utils)

        self.utils.logger.set_log_freq(len(self.dataloader.train))

    def run_job(self):
        self.utils.logger.log_console("Starting the job")
        exp_type = self.config["experiment_type"]
        if exp_type == "challenge":
            self.run_challenge_job()
        elif exp_type == "defense":
            self.run_defense_job()
        elif exp_type == "attack":
            self.run_attack_job()
        else:
            print("unknown experiment type")

    def run_defense_job(self):
        for epoch in range(self.config["total_epochs"]):
            print("starting epoch ", epoch)
            self.defense_train()
            self.defense_test()
            self.epoch_summary()
            print("finished epoch ", epoch)
        self.generate_challenge()

    def run_attack_job(self):
        print("running attack job")
        for epoch in range(self.config["total_epochs"]):
            self.attack_train()
            self.attack_test()
            self.epoch_summary()

    def run_challenge_job(self):
        self.utils.load_saved_models()
        self.generate_challenge()

    def defense_train(self) -> None:
        self.algo.train()
        self.model.train()
        for _, sample in enumerate(self.dataloader.train):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            items["server_grads"] = self.model.processing(z, items["pred_lbls"])
            self.algo.backward(items)
            # if _ == 10:
            #     return

    def defense_test(self) -> None:
        self.algo.eval()
        self.model.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.model.processing(z, items["pred_lbls"])

            # if _ == 10:
            #     return

    def attack_train(self) -> None:
        self.algo.train()
        return
        for _, sample in enumerate(self.dataloader.train):
            # print(_, sample)
            items = self.utils.get_data(sample)
            # print(items)
            z = self.algo.forward(items)
            self.algo.backward(items)

    def attack_test(self):
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.train):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            return

    def epoch_summary(self):
        self.utils.logger.flush_epoch()
        self.utils.save_models()

    def generate_challenge(self) -> None:
        challenge_dir = self.utils.make_challenge_dir(self.config["results_path"])
        self.algo.eval()
        loss = torch.nn.MSELoss()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            print("BRUH LOSSS: ", loss(z, z))
            print("BRUH LOSSS: ", loss(z, self.algo.forward(items)))
            self.utils.save_data(z, items["filename"], challenge_dir)
