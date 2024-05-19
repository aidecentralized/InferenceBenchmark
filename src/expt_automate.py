import sys
from scheduler import Scheduler
from utils.config_utils import combine_configs, process_config


def run_experiment(config):
    config = process_config(config)
    scheduler = Scheduler()
    scheduler.assign_config_by_dict(config)
    scheduler.initialize()
    scheduler.run_job()


"""bench_config = {
    "method": "disco",
    "client": {"model_name": "resnet18", "split_layer": 6,
               "pretrained": False, "optimizer": "adam", "lr": 3e-4,
               "pruning_ratio": 0.99, "pruning_style": "learnable", "alpha": 0.5,
               "grid_crop" : False,
               "proxy_adversary" : {"img_size": 128}},
    "server": {"model_name": "resnet18", "split_layer":6, "logits": 2, "pretrained": False,
               "lr": 3e-4, "optimizer": "adam"},
    "learning_rate": 0.01,
    "total_epochs": 150,
    "training_batch_size": 256,
    "dataset": "fairface",
    "protected_attribute": "data",
    "prediction_attribute": "gender",
    "img_size": 128,
    "split": False,
    "test_batch_size": 64,
    "exp_id": "1",
    "exp_keys": ["client.pruning_ratio", "client.pruning_style", "client.grid_crop"],
    "dataset_path": "/u/abhi24/Datasets/Faces/fairface/",
    "experiments_folder": "/u/abhi24/Workspace/simba/experiments/",
    "gpu_devices":[0,1]
}
hparam = {"pruning_ratio": [0.999, 0.9995, 0.9999]}
"""

bench_config = {
    "experiment_type": "attack",
    "method": "supervised_decoder",
    "adversary": {"loss_fn": "ssim", "img_size": 128,
                  "optimizer": "adam", "lr": 0.01, "attribute": "data"},
    "total_epochs": 100,
    "training_batch_size": 128,
    "challenge_experiment": "",
    "protected_attribute": "data",
    "dataset": "fairface",
    "exp_id": "1",
    "img_size": 128,
    "split": True,
    "train_split": 0.9,
    "test_batch_size": 64,
    "exp_keys": ["adversary.loss_fn"],
}
hparam = {"challenge_experiment": ["uniform_noise_fairface_data_resnet18_split6_1_distribution_gaussian_mean_0_sigma_300",
                                   "disco_fairface_data_resnet18_split6_1_pruning_ratio_0.9999_pruning_style_learnable_grid_crop_False",
                                   "disco_fairface_data_resnet18_split6_1_pruning_ratio_0.9_pruning_style_learnable_grid_crop_False",
                                   "nopeek_fairface_data_resnet18_split6_1_alpha_0.97",
                                   "uniform_noise_fairface_data_resnet18_split6_1_distribution_gaussian_mean_0_sigma_0"]}

sys_config = {
  "dataset_path": "/home/justinyu/fairface/",
  "experiments_folder": "/home/justinyu/experiments/",
  "gpu_devices": [1, 3]
}

# sys_config = {"dataset_path": "/u/abhi24/Datasets/Faces/fairface/",
#               "experiments_folder": "/u/abhi24/Workspace/simba/experiments/",
#               "gpu_devices":[0,1]}

bench_config = combine_configs(bench_config, sys_config)

for param in hparam.keys():
    for val in hparam[param]:
        # For obfuscation
        # bench_config["client"][param] = val

        # For attack
        bench_config[param] = val
        run_experiment(bench_config)

