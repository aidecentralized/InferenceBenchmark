import argparse
from main import run_experiment

parser = argparse.ArgumentParser(description='Process inputs for chap project.')

# Necessary inputs unless more advanced uses are being done

parser.add_argument('--dataset', type=str, help="name of the dataset to train or use for classification. Ex. Fairface (must follow the guidelines for the organization in order to be used)")
parser.add_argument('--split_layer', type=int, default=4, help="number representing where to split the model to be used for chap.")
parser.add_argument('--dataset_path', type=str, help="path on device where dataset resides. Ex. /u/user/")
parser.add_argument('--prediction_attribute', type=str, default="gender", help="name of label attribute to protect")
parser.add_argument('--protected_attribute', type=str, default="race", help="name of label attribute to protect")

# check logic for experiment paths (new, etc. for full functionality)
parser.add_argument('--experiment_path', type=str, default=None, help="path to store and access experiements.")

# Update help
parser.add_argument('--pruning_style', type=str, default="network", help="style of pruning for data obsufication. Supported styles: 'network', 'nopruning', 'adversarial', 'maxentropy', 'noise'")

# More advanced parameters
parser.add_argument('--batch_size', type=int, default=64, help="number of samples per training / testing batch.")
parser.add_argument('--pruning_ratio', type=float, default=0.65, help="number representing the pruning ratio for the chap experiment.")
parser.add_argument('--total_epochs', type=int, default=140, help="number representing the amount of epochs desired for training.")
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate used to train model chap models. Ex. 0.01")
parser.add_argument('--dir_of_inputs', type=str, default=None, help="Directory of inputs to be classified.")

if __name__ == '__main__':
    args = parser.parse_args()
    new_args = args.__dict__
    run_experiment(False, new_args)