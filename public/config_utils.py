import os
import json


def load_config_as_dict():
    """
    Load the contents of the config file as a dict object
    """
    base_path = os.path.dirname(__file__)
    rel_path = "config.json"
    path = os.path.join(base_path, rel_path)
    json_dict = None

    with open(path) as json_file:
        json_dict = json.load(json_file)

    return json_dict


def load_config():
    """
    Properly load and configure the json object into the expected config format. This includes
    the calculations of dynamic variables.
    """
    json_dict = load_config_as_dict()
    experiment_dict = json_dict.get("experiment_config", {}).copy()
    research_dict = json_dict.get("research_config", {}).copy()
    research_dict.update(experiment_dict)
    json_dict = research_dict

    json_dict['num_gpus'] = len(json_dict.get('gpu_devices'))
    json_dict['train_batch_size'] = json_dict.get('train_batch_size', 64) * json_dict['num_gpus']

    if 'is_grid_crop' not in json_dict:
        json_dict['is_grid_crop'] = False

    if 'manual_expt_name' in json_dict.keys():
        '''serves two use cases - generating challenge for past experiments which followed different
        naming convention. The other case is when we want to transfer a pretrained pruner network to
        a different client model.'''
        experiment_name = json_dict['manual_expt_name'] 
    else:
        if json_dict['is_grid_crop']:
            grid_crop = "grid_crop"
        else:
            grid_crop = ""
        experiment_name = "pruning_{}_{}_{}_split{}_ratio{}_{}_{}_{}".format(json_dict['pruning_style'],
            json_dict['dataset'], json_dict['model_name'], json_dict['split_layer'],
            json_dict['pruning_ratio'], json_dict['protected_attribute'], grid_crop, json_dict['exp_id'])

    experiments_folder = json_dict["experiments_folder"]
    results_path = experiments_folder + experiment_name
    
    log_path = results_path + "/logs/"
    model_path = results_path + "/saved_models"
    
    json_dict["experiment_name"] = experiment_name
    json_dict["log_path"] = log_path
    json_dict["model_path"] = model_path
    json_dict["results_path"] = results_path

    return json_dict

def update_config(config=None, new_variables=None):
    """
    Process as to update the pruning ratio with a new given parameters mapping their config names to their new values.
    If None are given, the old config remains unchanged.
    """
    json_dict = load_config_as_dict()
    experiment_dict = json_dict.get("experiment_config", {}).copy()
    research_dict = json_dict.get("research_config", {}).copy()

    if type(new_variables) is not dict:
        print("New configuration variables not given as dictionary. Type needs to be dictionary.")
    else:
        for key in new_variables.keys():
            if key in experiment_dict:
                experiment_dict[key] = new_variables[key]
            if key in research_dict:
                experiment_dict[key] = new_variables[key]
    json_dict["experiment_config"] = experiment_dict
    json_dict["research_config"] = research_dict

    base_path = os.path.dirname(__file__)
    rel_path = "config.json"
    path = os.path.join(base_path, rel_path)
    with open(path, 'w') as fp:
        json.dump(json_dict, fp, sort_keys=True, indent=4)


def load_experiment_config(experiment_path):
    """
    Properly load and configure the json object into the expected config format. This includes
    the calculations of dynamic variables.
    """
    json_dict = load_relative_config_as_dict(experiment_path)
    experiment_dict = json_dict.get("experiment_config", {}).copy()
    research_dict = json_dict.get("research_config", {}).copy()
    research_dict.update(experiment_dict)
    json_dict = research_dict

    json_dict['num_gpus'] = len(json_dict.get('gpu_devices'))
    json_dict['train_batch_size'] = 64 * json_dict['num_gpus']

    experiment_name = "pruning_{}_{}_{}_split{}_ratio{}_{}".format(json_dict['pruning_style'],
        json_dict['dataset'], json_dict['model_name'], json_dict['split_layer'],
        json_dict['pruning_ratio'], json_dict['exp_id'])

    results_path = "../experiments/" + experiment_name
    log_path = results_path + "/logs/"
    model_path = results_path + "/saved_models"
    
    json_dict["experiment_name"] = experiment_name
    json_dict["log_path"] = log_path
    json_dict["model_path"] = model_path
    json_dict["results_path"] = results_path

    # Inputs default values for missing / old parameters if not present

    needed_params = {
    "pruner_hparams": {"needed": ["split_layer","pruning_ratio","pruning_style"],"pretrained": False},
    "server_hparams": {"logits": 2,"needed": ["split_layer"],"pretrained": False},
    "adversary_hparams": {"logits": 7,"needed": ["split_layer"],"pretrained": False},
    "client_hparams": {"needed": ["split_layer", "is_grid_crop"],"pretrained": False}
    }

    for np in needed_params.keys():
        if np not in json_dict:
            json_dict[np] = needed_params[np]

    return json_dict

def load_relative_config_as_dict(experiment_path):
    """
    Load the contents of the config file as a dict object
    """
    base_path = os.path.dirname(experiment_path)
    rel_path = "config.json"
    path = os.path.join(base_path, rel_path)
    json_dict = None

    with open(path) as json_file:
        json_dict = json.load(json_file)

    return json_dict


def save_dict_as_json(dictionary, new_file_path):
    with open(new_file_path, "w") as outfile: 
        json.dump(dictionary, outfile) 

