"""
This is meant to run all experiment models on every photo used for training and evaluation. The purpose of this is to compare which outcomes have been correctly classified to determine any potential underlying biases and have a better idea of the model's overall performance.
"""

import os
import torch
import config_utils as utils
import models
from collections import OrderedDict 
from dataset_utils import Custom
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import OrderedDict
from adv_models import AdversaryModelGen, AdversaryModelPred, reconstruction_loss

gpu_id = 0

def load_client_model(models_path, config):
    """
    Returns Pytorch client model loaded given the client model's path
    """
    device = load_device()
    client_hparams = config.get("client_hparams")
    for needed_param in client_hparams.get("needed", []):
        client_hparams[needed_param] = config.get(needed_param)
    model_file = os.path.join(models_path, "client_model.pt")
    model = models.ResNet18Client(client_hparams)
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except:
        model = models.ResNet18Client(client_hparams)
        state_dict = cleanse_state_dict(torch.load(model_file, map_location=device))
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model

def load_server_model(models_path, config):
    """
    Returns Pytorch model loaded given the model's name
    """
    device = load_device()
    server_hparams = config.get("server_hparams")
    for needed_param in server_hparams.get("needed", []):
        server_hparams[needed_param] = config.get(needed_param)
    model_file = os.path.join(models_path, "server_model.pt")
    model = models.ResNet18Server(server_hparams)
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except:
        model = models.ResNet18Server(server_hparams)
        state_dict = cleanse_state_dict(torch.load(model_file, map_location=device))
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model

def load_adversary_model(models_path, config, client_model):
    """
    Returns Pytorch model loaded given the model's name
    """
    device = load_device()
    protected_attribute = config.get("protected_attribute")
    adversary_hparams = config.get("adversary_hparams")
    adversary_hparams['split_layer'] = config.get('split_layer')
    client_channels = client_model(torch.rand(1, 3, 128, 128).to(device)).shape[1]
    if protected_attribute == "data":
        adversary_hparams.update({"channels": client_channels})
        model = AdversaryModelGen(adversary_hparams)
    else:
        model = AdversaryModelPred(adversary_hparams)
    for needed_param in adversary_hparams.get("needed", []):
        adversary_hparams[needed_param] = config.get(needed_param)
    model_file = os.path.join(models_path, "adversary_model.pt")
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except:
        if protected_attribute == "data":
            adversary_hparams.update({"channels": client_channels})
            model = AdversaryModelGen(adversary_hparams)
        else:
            model = AdversaryModelPred(adversary_hparams)
        state_dict = cleanse_state_dict(torch.load(model_file, map_location=device))
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model

def load_pruning_model(models_path, config, client_model):
    """
    Returns Pytorch model loaded given the model's name
    """
    device = load_device()
    pruner_hparams = config.get("pruner_hparams")
    client_channels = list(client_model.model.children())[config.get('split_layer')][-1].bn2.num_features
    pruner_hparams.update({"logits": client_channels})
    for needed_param in pruner_hparams.get("needed", []):
        pruner_hparams[needed_param] = config.get(needed_param)
    model_file = os.path.join(models_path, "pruner_model.pt")
    model = models.PruningNetwork(pruner_hparams)
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
    except:
        model = models.PruningNetwork(pruner_hparams)
        state_dict = cleanse_state_dict(torch.load(model_file, map_location=device))
        model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model.cuda()
    return model

def load_all_saved_models_paths():
    """
    Attempts to load every experiment path for every completed model
    """
    chap_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    experiment_path = os.path.join(chap_path, "experiments")
    all_contents = os.listdir(experiment_path)
    experiment_paths = [os.path.join(experiment_path, path) for path in all_contents]
    saved_models_paths = [os.path.join(path, "saved_models") for path in experiment_paths]
    return saved_models_paths

def get_all_models(saved_models_path, config):
    models = dict()
    models["adversary_model"] = load_adversary_model(saved_models_path, config)
    models["client_model"] = load_client_model(saved_models_path, config)
    models["pruner_model"] = load_pruning_model(saved_models_path, config, models['client_model'])
    models["server_model"] = load_server_model(saved_models_path, config)
    return models

def load_all_models(specific_experiments=None):
    config = utils.load_config()
    all_saved_models_paths = load_all_saved_models_paths()
    if specific_experiments:
        all_saved_models_paths = [path for path in all_saved_models_paths if any([experiment in path for experiment in specific_experiments])]
    d = dict()
    for p in all_saved_models_paths:
        try:
            rel_config = utils.load_experiment_config(p)
            models_dict = get_all_models(p, rel_config)
            d[p] = models_dict
        except:
            pass # this is the case where directory waas created, but the models were not saved properly or the experiment failed to run
    return d

def load_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    return device

def load_image_as_tensor(filepath, config):
    img = Image.open(filepath)
    transforms = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()])
    img = transforms(img)
    return img

def eval_mode(client_model, server_model, adversary_model):
    """
    determine which models to set into eval_mode
    """
    client_model.eval()
    server_model.eval()
    adversary_model.eval()

def classify_directory_data(image_set_directory, dataset_config=None, specific_experiments=None):
    """
    Meant to evaluate the data of the image_set_directory per every model saved under the experiments path.
    Specific_experiments is to run this code only for experiments of interest rather than all experiments. List of exact experiments.
    If dataset other than Custom is used, must send custom dataset_config.
    Reference code below.
    """
    
    config = utils.load_config()
    device = load_device()
    IM_SIZE = config["img_size"]
    imageTransform = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor()])
    if config["dataset"] == "fairface":
        dataset_config = {"transforms": imageTransform,
                   "train": False,
                   "path": image_set_directory,
                   "prediction_attribute": config["prediction_attribute"],
                   "protected_attribute": config["protected_attribute"],
                   "format": "jpg"}
    dataset = Custom(dataset_config)
    imageLoader = torch.utils.data.DataLoader(
        dataset, batch_size=config.get("test_batch_size"),
        shuffle=False, num_workers=5)

    all_saved_model_sets = load_all_models(specific_experiments)
    all_model_sets_classifications = dict()
    for name_of_model_set, model_set in all_saved_model_sets.items():
        total, task_pred_correct, privacy_pred_correct = 0, 0, 0
        model_set_classification = dict()
        client_model = model_set['client_model']
        server_model = model_set['server_model']
        adversary_model = model_set['adversary_model']
        pruner_model = model_set['pruner_model']
        eval_mode(client_model, server_model, adversary_model)
        with torch.no_grad():
            for batch_idx, sample in enumerate(imageLoader):
                filepath = sample["filepath"]
                data = Variable(sample["img"]).to(device)
                prediction_labels = Variable(sample["prediction_label"]).to(device)
                protected_labels = Variable(sample["private_label"]).to(device)

                z = client_model(data)
                z_hat, indices = pruner_model(z)

                task_attribute_prediction = server_model(z_hat)
                protected_attribute_prediction = adversary_model(z_hat)
                task_attribute_predicted = task_attribute_prediction.argmax(dim=1)
                privacy_attribute_predicted = protected_attribute_prediction.argmax(dim=1)
                num_of_files = len(filepath)
                for i in range(len(filepath)):
                    name = filepath[i].split("/")[-1]
                    model_set_classification[name] = {
                        "task_attribute_prediction": task_attribute_predicted[i].item(),
                        "privacy_attribute_predicted": privacy_attribute_predicted[i].item()
                    }
                task_pred_correct += (task_attribute_prediction.argmax(dim=1) ==
                                  prediction_labels).sum().item()
                privacy_pred_correct += (protected_attribute_prediction.argmax(dim=1) ==
                                     protected_labels).sum().item()
                # UNcomment if known csv file to know score, etc.
                # total += int(data.shape[0])
                # model_set_classification["task_pred_correct"] = task_pred_correct
                # model_set_classification["privacy_pred_correct"] = privacy_pred_correct
                # model_set_classification["total"] = total
        all_model_sets_classifications[name_of_model_set] = model_set_classification

    return all_model_sets_classifications


def cleanse_state_dict(state_dict):
    """
    This is an mismatch of expected keys, etc. Issua comes up is saved via gpu, but running on cpu, etc.
    Ex. mismatch: keys not matching
    Expecting: {"model.0.weight", ...}
    Received: {"module.model.0.weight", ...}
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    return new_state_dict


def uncleanse_state_dict(state_dict):
    """
    This is an mismatch of expected keys, etc. Issua comes up is saved via cpu, but running on gpu, etc.
    Ex. mismatch: keys not matching
    Expecting: {"module.model.0.weight", ...}
    Received: {"model.0.weight", ...}
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        nk = "module." + k
        new_state_dict[nk] = v
    return new_state_dict


def load_all_saved_models_paths_to_experiment_name():
    """
    Attempts to load every experiment path for every completed model
    """
    d = dict()
    chap_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    experiment_path = os.path.join(chap_path, "experiments")
    all_contents = os.listdir(experiment_path)
    for path in all_contents:
        d[os.path.join(os.path.join(experiment_path, path), "saved_models")] = path
    return d


def save_each_classification(all_model_sets_classifications, base_path_for_new_files):
    if not os.path.exists(base_path_for_new_files):
        os.makedirs(base_path_for_new_files)
    key_to_experiment_name_dict = load_all_saved_models_paths_to_experiment_name()
    for experiment_path, experiment_outcomes in all_model_sets_classifications.items():
        new_file_suffix = key_to_experiment_name_dict[experiment_path] + ".json"
        new_file = os.path.join(base_path_for_new_files, new_file_suffix)
        utils.save_dict_as_json(experiment_outcomes, new_file)

def load_models(experiment_path):
    config = utils.load_experiment_config(experiment_path)
    
    split_layer = get_split_layer(experiment_path)
    config['split_layer'] = split_layer
    # config = c.load_config()
    base_p = experiment_path # '/Users/ethangarza/exp_name/'
    ml1_path = base_p + "saved_models/"
    ml2_path = base_p  + "saved_models/"
    server_model = load_server_model(ml1_path, config)
    client_model = load_client_model(ml2_path, config)
    try:
        adversary_model = load_adversary_model(ml1_path, config, client_model)
    except:
        adversary_model = None # means that using outdated adversary model architect
    try:
        pruner_model = load_pruning_model(ml1_path, config, client_model)
    except:
        pruner_model = None  # means using outdated pruner model. need to replace pruning model architect with specific model from experiment
    return server_model, client_model, pruner_model, adversary_model


def get_split_layer(experiment_path):
    config = utils.load_experiment_config(experiment_path)
    return config.get("split_layer")


def get_config(experiment_path):
    config = utils.load_experiment_config(experiment_path)
    return config


def load_last_epoch(experiment_path):
    log_dir = os.path.join(experiment_path, "logs")
    last_epoch = 0
    for f in os.listdir(log_dir):
        try:
            local_epoch = int(f)
        except:
            local_epoch = 0
        if local_epoch > last_epoch:
            last_epoch = local_epoch
    return last_epoch

def make_flat_model(model):
    """
    Translates the same weights in the same order into a new model
    without any encapuslated modules
    Inputs: Pytorch model
    Outputs: Sequential Pytorch model
    """
    flattened_model_as_list = flatten_model_as_list(model)
    od = OrderedDict()
    for i in range(len(flattened_model_as_list)):
        od[str(i)] = flattened_model_as_list[i]
    flattened_model = torch.nn.Sequential(od)
    return flattened_model

def flatten_model_as_list(model):
    """
    Returns a list of submodules from a given Pytorch model
    """
    current_modules = []
    if len(model._modules) > 0:
        queue = model._modules
        while len(queue) > 0:
            sub_model = queue.popitem(False)[1]
            current_modules = current_modules + flatten_model_as_list(sub_model)
    else:
        current_modules.append(model)
    return current_modules

if __name__ == '__main__':
    # Meant to serve as an example of how to use function(s). Note optional arguemnts
    # specific_experiments = ['pruning_network_fairface_resnet18_scratch_split6_ratio0.2_1']
    specific_experiments=None
    image_set_directory = '/Users/ethangarza/FairFace/fairface-img-margin025-trainval/val'
    all_model_sets_classifications = classify_directory_data(image_set_directory, specific_experiments=specific_experiments)
    base_path_for_new_files = '/Users/ethangarza/FairFace/fairface-img-margin025-trainval/val_outcome'
    save_each_classification(all_model_sets_classifications, base_path_for_new_files)
    pass
