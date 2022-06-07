from scheduler import Scheduler
from utils.config_utils import process_config

bench_config = {
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
pruning_ratios = [0.999, 0.9995, 0.9999]
for ratio in pruning_ratios:
    bench_config["client"]["pruning_ratio"] = ratio

    bench_config = process_config(bench_config)
    scheduler = Scheduler()
    scheduler.assign_config_by_dict(bench_config)
    scheduler.initialize()
    scheduler.run_job()
