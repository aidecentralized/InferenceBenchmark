
import json
import os
import random

ALL_TOKEN = "@all"

def apply_attacks_settings(curr_attacks: set[str], settings, all_attacks: set[str]):
    include, exclude = set(settings["include"]), set(settings["exclude"])
    new_attacks = curr_attacks.copy()
    new_attacks = new_attacks.union(all_attacks) if ALL_TOKEN in include else new_attacks.union(include)
    new_attacks = set() if ALL_TOKEN in exclude else new_attacks.difference(exclude)

    assert len(new_attacks.difference(all_attacks)) == 0 # check attacks in all_attacks

    return new_attacks

# replace this
def get_expname(model_config, group_id=""):
    return f"{model_config['method']}-{group_id}-{random.randint(0,1000)}"

def create_configs_for_bench(all_bench_config):
    defense_base_config_paths, attack_base_config_paths = get_base_config_paths(all_bench_config)
    # add error catching here
    defense_base_configs = {name: json.load(open(path, 'r')) for name, path in defense_base_config_paths.items()}
    attack_base_configs = {name: json.load(open(path, 'r'))  for name, path in attack_base_config_paths.items()}

    all_attacks = set(attack_base_configs.keys())
    default_included_attacks = apply_attacks_settings(set(), all_bench_config["default_attacks"], all_attacks)

    attack_to_defense_map = {attack: set() for attack in all_attacks}

    all_defense_configs = []
    for defense_name, model_bench_config in all_bench_config["defense_models"].items():
        base_config = defense_base_configs[defense_name]
        run_groups = model_bench_config["run_groups"].items()
        model_included_attacks = apply_attacks_settings(default_included_attacks, model_bench_config["attacks"], all_attacks) if "attacks" in model_bench_config else default_included_attacks

        if len(run_groups) == 0:
            all_defense_configs.append(base_config)
            [attack_to_defense_map[attack].add(get_expname(base_config)) for attack in model_included_attacks]
            continue

        for group_id, run_group in model_bench_config["run_groups"].items():
            hparams = run_group["hparams"]
            hparam_combos = generate_hparam_combos_from_hparams(hparams)
            new_configs = generate_hparams_configs(base_config, hparam_combos, group_id)
            all_defense_configs.extend(new_configs)

            group_included_attacks = apply_attacks_settings(model_included_attacks, run_group["attacks"], all_attacks) if "attacks" in run_group else model_included_attacks
            [[attack_to_defense_map[attack].add(get_expname(config,group_id)) for attack in group_included_attacks] for config in new_configs]

    all_attack_configs = []
    for attack_name, model_bench_config in all_bench_config["attack_models"].items():
        base_config = attack_base_configs[attack_name]
        run_groups = model_bench_config["run_groups"].items()

        attack_configs = []
        if len(run_groups) == 0:
            attack_configs.append(base_config)
        else:
            for group_id, run_group in model_bench_config["run_groups"].items():
                hparams = run_group["hparams"]
                hparam_combos = generate_hparam_combos_from_hparams(hparams)
                new_configs = generate_hparams_configs(base_config, hparam_combos, group_id)
                attack_configs.extend(new_configs)

        defenses_to_attack = attack_to_defense_map[attack_name]
        for config in attack_configs:
            for defense in defenses_to_attack:
                jmespath_update("challenge_experiment", defense, config)
                all_attack_configs.append(config)
    return all_defense_configs, all_attack_configs


def generate_hparams_configs(base_config, hparam_runs, group_id):
    new_configs = [base_config]
    for hparam_dict in hparam_runs:
        new_config = base_config.copy()
        for hparam_path, val in hparam_dict.items():
            jmespath_update(hparam_path, val, new_config)
        jmespath_update("rungroup_id", group_id, new_config)
        new_configs.append(new_config)
    return new_configs

"""
In-place modifies source dict with keypath and val
"""
def jmespath_update(key, val, source):
    curr_key, *rest = key.split(".")
    if len(rest) > 0:
        if curr_key not in source:
            source[curr_key] = {}
        return jmespath_update(".".join(rest), val, source[curr_key])
    source[curr_key] = val

"""
Creates list of hparam combinations from a single model bench config
"""
def generate_hparam_combos_from_hparams(hparams):
    def recurse(flattened_hparams, runs = [{}]):
        if len(flattened_hparams) == 0:
            return runs
        hparam, values = flattened_hparams[0]
        new_runs = []
        for run in runs:
            new_runs.extend([dict({hparam: val}, **run) for val in values])
        return recurse(flattened_hparams[1:], new_runs)
    result = recurse([[key, val] for key, val in hparams.items()])
    return [item for item in result if item] # exclude empty dict
    



def get_base_config_paths(bench_config) -> tuple[dict[str, str], dict[str,str]]:
    config_folder = bench_config["config_path"]

    defenses: dict[str, str] = {}
    attacks: dict[str, str] = {}
    for name, model_def in bench_config["defense_models"].items():
        filename = model_def["config"] if "config" in model_def else f"{name}.json"
        filepath = os.path.join(config_folder, filename)
        if os.path.isfile(filepath):
            defenses[name] = filepath
        else:
            print(f"No config found for defense model '{name}'. Tried path: {filepath}")
    
    for name, model_def in bench_config["attack_models"].items():
        filename = model_def["config"] if "config" in model_def else f"{name}.json"
        filepath = os.path.join(config_folder, filename)
        if os.path.isfile(filepath):
            attacks[name] = filepath
        else:
            print(f"No config found for attack model '{name}'. Tried path: {filepath}")
    
    return defenses, attacks