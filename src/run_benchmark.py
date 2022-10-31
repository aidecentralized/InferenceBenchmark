from pprint import pprint
import argparse
from utils.benchmark_utils import create_configs_for_bench

# bench_default = "./benchmark.json"

# parser = argparse.ArgumentParser(description='Run SIMBA benchmark script')
# parser.add_argument('benchmark_config', default=bench_default, type=open,
#                     help='filepath for benchmark config, default: {}'.format(bench_default))
# # not implemented
# parser.add_argument('-f', '--force', action='store_true',
#                     help='force rerun all benchmarks, even if they exist')

# args = parser.parse_args()


"""
Run Groups
For each model, you can specify any number of 'run groups' with a unique group id. Each run group allows you to specific a set of hyperparams to run the model with. 
You can also include/exclude attacks, customize if they will be shown on the final benchmark graph etc. 

Hyperparameters
Each hparams config is a dictionary where keys are json access paths in the base config of the model, and the value is an array of potential hparam values.
The cartesian product of hparam values is calculated and a model will be run on each. For K hparams, we will run N_0 * ... * N_K models where N_i is the length
of the values array for hparam i.

Attacks:
Which attacks to run on which defense can be configured using attack settings. These consist of "include", "exclude" and a special "@all" token.
Attacks can be included/excluded globally, per model, and per run group


Benchmark Config Fields:

{
config_path: path to base folder for config files
defense_models/attack_models: configs for each defense/attack model
    [model name]
        attacks: (optional, configure attacks to be run on this model)
        run_groups: (see Run Groups above)
            [group id]
                hparams: (see Hyperparameters above)
                attacks: (optional, configue attacks to be run on this group)
default_attacks: settings for which attacks will be default be run on all defense models
}



TODOS:
- will need to change experiment naming system
- what to do with attack hp? should run all attacks variations on every defense?
- option to only run attack on best performing defense
"""

bench_config = {
    "config_path": "./configs",
    "defense_models": {
        "cloak": {
            "run_groups": {
                "run1": {
                    # add option to correlate params tgt, i.e. split layer
                    "hparams": {
                        "client.min_scale": [0, 1],
                        "server.lr": [3e-4, 3e-3],
                    },
                    "attacks": {"include": [], "exclude": []}
                },
                "run2": {
                    # add option to correlate params tgt, i.e. split layer
                    "hparams": {},
                    "attacks": {"include": ["input_optimization_attack"], "exclude": []}
                }
            },
            "attacks": {
                "include": [],
                "exclude": [],
            }
        }
    },
    "attack_models": {
        "supervised_decoder": {
            "config": "decoder_attack.json",
            "run_groups": {},
        },
        "input_optimization_attack": {
            "run_groups": {},
        }
    },
    "default_attacks": {
        "include": ["supervised_decoder"],
        "exclude": [],
    }
}


if __name__ == '__main__':
    pprint(create_configs_for_bench(bench_config))