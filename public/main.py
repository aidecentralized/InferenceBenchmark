import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import shutil
from config_utils import load_config

from scheduler import Scheduler
from models import ResNet18Client, ResNet18Server, PruningNetwork, loss_fn
from adv_models import AdversaryModelGen, AdversaryModelPred, reconstruction_loss
from utils import copy_source_code
import model_utils
import config_utils
import dataset_utils
import model_utils as m

def update_args(config, new_args):
    config.update(new_args)
    server_logits, adv_logits = dataset_utils.infer_data_info(new_args.get('dataset_path'), new_args.get("dataset"), config.get("protected_attribute"), config.get("prediction_attribute"))
    config["server_logits"] = server_logits
    config["adversary_logits"] = adv_logits
    config_utils.update_config(new_variables=new_args)


def run_experiment(auto=None, new_args=None):
    config = load_config()

    if new_args:
        update_args(config, new_args)
        print("updated config:", config)

    experiment_type = config.get("experiment_type", "challenge")
    gpu_devices = config.get("gpu_devices")
    pruning_ratio = config.get("pruning_ratio", 0.6)
    split_layer = config.get("split_layer", 4)
    lr = config.get("learning_rate", 0.01)
    is_grid_crop = config.get("is_grid_crop", False)
    pruning_style = config.get("pruning_style", "nopruning")
    results_path = config.get("results_path")
    protected_attribute = config.get("protected_attribute")
    adversary_logits = config.get("adversary_logits", 10)
    server_logits = config.get("server_logits", 2)
    client_pretrained = config.get("client_pretrained", False)
    adversary_pretrained = config.get("adversary_pretrained", False)
    server_pretrained = config.get("server_pretrained", False)
    pruner_pretrained = config.get("pruner_pretrained", False)

    gpu_devices = config.get("gpu_devices")
    gpu_id = gpu_devices[0]
    gpu_available = False

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
        gpu_available = True
    else:
        device = torch.device('cpu')

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    client_hparams = {"pretrained": client_pretrained, "split_layer": split_layer, "is_grid_crop": is_grid_crop}
    server_hparams = {"logits": server_logits, "pretrained": server_pretrained, "split_layer": split_layer}
    adversary_hparams = {"logits": adversary_logits, "pretrained": adversary_pretrained, "split_layer": split_layer}
    pruner_hparams = {"pretrained": pruner_pretrained, "split_layer": split_layer,
                      "pruning_ratio": pruning_ratio, "pruning_style": pruning_style}

    if experiment_type == "challenge":
        challenge_path = results_path + "/challenge/"
        if os.path.isdir(challenge_path):
            if auto:  # This it to not duplicate work already done and to continue running experiments
                return None
            print("Challenge for experiment {} already present".format(config.get("experiment_name")))
            inp = input("Press e to exit, r to replace it: ")
            if inp == "e":
                exit()
            elif inp == "r":
                shutil.rmtree(results_path + "/challenge/")
            else:
                print("Input not understood")
                exit()

    else:
        experiment_path = None
        print("exp path:", results_path)
        print("name:", results_path)
        if os.path.isdir(results_path):
            if auto:  # This it to not duplicate work already done and to continue running experiments
                print("silently skipping experiment", config.get('experiment_name'))
                return None
            print("Experiment {} already present".format(config.get("experiment_name")))
            inp = input("Press e to exit, r to replace it, c to continue training: ")
            if inp == "e":
                exit()
            elif inp == "r":
                shutil.rmtree(results_path)
            elif inp == "c":
                experiment_path = results_path
            else:
                print("Input not understood")
                exit()
        if not experiment_path:
            copy_source_code(results_path)
            os.mkdir(config.get('model_path'))
            os.mkdir(config.get('log_path'))


    if experiment_path:
        try:
            server_model, client_model, pruner, adversary_model = model_utils.load_models(experiment_path)
        except:
            server_model, client_model, pruner, adversary_model = model_utils.load_models(experiment_path+ "/")
    else:
        client_model = ResNet18Client(client_hparams).to(device)
        server_model = ResNet18Server(server_hparams).to(device)

    client_channels = client_model(torch.rand(1, 3, config.get('img_size'), config.get('img_size')).to(device)).shape[1]
    #client_channels = list(client_model.model.children())[split_layer][-1].bn2.num_features
    if protected_attribute == "data":
        adversary_hparams.update({"channels": client_channels})
        if not experiment_path:
            adversary_model = AdversaryModelGen(adversary_hparams).to(device)
        privacy_loss_fn = reconstruction_loss
    else:
        if not experiment_path:
            adversary_model = AdversaryModelPred(adversary_hparams).to(device)
        privacy_loss_fn = loss_fn

    pruner_hparams.update({"logits": client_channels})
    if not experiment_path:
        pruner = PruningNetwork(pruner_hparams).to(device)

    prediction_loss_fn = loss_fn

    if config.get("optim_type") == 'adam':
        client_optimizer = optim.Adam(client_model.parameters(), lr=lr)
        server_optimizer = optim.Adam(server_model.parameters(), lr=lr)
        adversary_optimizer = optim.Adam(adversary_model.parameters(), lr=lr)
        pruner_optimizer = optim.Adam(pruner.parameters(), lr=lr)
    else:
        client_optimizer = optim.SGD(client_model.parameters(), lr=lr)
        server_optimizer = optim.SGD(server_model.parameters(), lr=lr)
        adversary_optimizer = optim.SGD(adversary_model.parameters(), lr=lr)
        pruner_optimizer = optim.SGD(pruner.parameters(), lr=lr)

    objects = {"client_model": client_model,
               "server_model": server_model,
               "adversary_model": adversary_model,
               "pruner": pruner,
               "client_optim": client_optimizer,
               "server_optim": server_optimizer,
               "adversary_optim": adversary_optimizer,
               "pruner_optim": pruner_optimizer,
               "prediction_loss_fn": prediction_loss_fn,
               "privacy_loss_fn": privacy_loss_fn}
    if gpu_available:
      objects["device"] = gpu_id
    else:
      objects["device"] = device

    last_epoch = None
    if experiment_path:
        last_epoch = m.load_last_epoch(experiment_path)

    print("last epoch:", last_epoch)

    scheduler = Scheduler(config, objects, last_epoch)

    
    if experiment_type == "training": 
        # call the training loop
        num_times_to_run = config.get("total_epochs", 0)
        if last_epoch:
            num_times_to_run = num_times_to_run - abs(last_epoch)
            if num_times_to_run < 0:
                num_times_to_run = 0
        print("times to run:", num_times_to_run)
        for epoch in range(num_times_to_run):
            scheduler.train()
            scheduler.test()
        scheduler.generate_challenge()
    elif experiment_type == "challenge":
        print("Creating challenge")
        scheduler.generate_challenge()
    else:
        print("Unknown experiment type")


if __name__ == '__main__':
    run_experiment()
