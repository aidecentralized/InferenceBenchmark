from interface import (load_config, load_algo, load_data, load_model, load_utils)

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
            self.defense_train()
            self.defense_test()
            self.epoch_summary()
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
            data = self.model.forward(z)
            items["decoder_grads"] = self.algo.infer(data,items["pred_lbls"])
            items["server_grads"] = self.model.backward(items["pred_lbls"],items["decoder_grads"])
            self.algo.backward(items)

    def defense_test(self) -> None:
        self.algo.eval()
        self.model.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            data = self.model.forward(z)
            self.algo.infer(data,items["pred_lbls"])
            self.model.compute_loss(data,items["pred_lbls"])

    def attack_train(self) -> None:
        if "no_train" in self.config and self.config["no_train"]:
            return
        self.algo.train()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.algo.backward(items)

    def attack_test(self):
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_images(z,items["filename"])

    def epoch_summary(self):
        self.utils.logger.flush_epoch()
        self.utils.save_models()

    def generate_challenge(self) -> None:
        challenge_dir = self.utils.make_challenge_dir(self.config["results_path"])
        self.algo.eval()
        for _, sample in enumerate(self.dataloader.test):
            items = self.utils.get_data(sample)
            z = self.algo.forward(items)
            self.utils.save_data(z, items["filename"], challenge_dir)
