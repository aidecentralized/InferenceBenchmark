import os
import json
import jmespath


def load_config_as_dict(filepaths):
    """
    Load the contents of the config file as a dict object
    """
    b_json = json.load(filepaths.b)
    s_json = json.load(filepaths.s)

    filepaths.b.close()
    filepaths.s.close()
    return b_json, s_json


def config_loader(filepath):
    """
    Properly load and configure the json object into the expected config format. This includes
    the calculations of dynamic variables.
    """
    b_json, s_json = load_config_as_dict(filepath)
    experiment_dict = s_json
    research_dict = b_json
    research_dict.update(experiment_dict)
    json_dict = research_dict

    json_dict['num_gpus'] = len(json_dict.get('gpu_devices'))
    json_dict['train_batch_size'] = json_dict.get('train_batch_size', 64) * json_dict['num_gpus']
    json_dict['experiment_type'] = json_dict.get('experiment_type') or "defense"

    if 'manual_expt_name' in json_dict.keys():
        '''serves two use cases - generating challenge for past experiments which followed different
        naming convention. The other case is when we want to transfer a pretrained pruner network to
        a different client model.'''
        experiment_name = json_dict['manual_expt_name']
    elif json_dict["experiment_type"] in ["defense", "challenge"]:
        experiment_name = "{}_{}_{}_{}_split{}_{}".format(
            json_dict['method'],
            json_dict['dataset'],
            json_dict['protected_attribute'],
            json_dict['client']['model_name'],
            json_dict['client']['split_layer'],
            json_dict['exp_id'])
        for exp_key in json_dict["exp_keys"]:
            item = jmespath.search(exp_key, json_dict)
            assert item is not None
            key = exp_key.split(".")[-1]
            assert key is not None
            experiment_name += "_{}_{}".format(key, item)
    else:
        assert json_dict['experiment_type'] == "attack"
        experiment_name = "{}_{}_{}".format(
            json_dict['method'],
            json_dict['challenge_experiment'],
            json_dict['exp_id'])
        for exp_key in json_dict["exp_keys"]:
            item = jmespath.search(exp_key, json_dict)
            assert item is not None
            key = exp_key.split(".")[-1]
            assert key is not None
            experiment_name += "_{}_{}".format(key, item)
        json_dict['challenge_dir'] = json_dict['experiments_folder'] +\
                                     json_dict['challenge_experiment'] +\
                                     "/challenge/"

    experiments_folder = json_dict["experiments_folder"]
    results_path = experiments_folder + experiment_name

    log_path = results_path + "/logs/"
    challenge_log_path = results_path + "/challenge-logs/"
    model_path = results_path + "/saved_models/"

    json_dict["experiment_name"] = experiment_name
    json_dict["log_path"] = log_path
    json_dict["challenge_log_path"] = challenge_log_path
    json_dict["model_path"] = model_path
    json_dict["results_path"] = results_path

    return json_dict


def update_config(new_variables=None):
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
