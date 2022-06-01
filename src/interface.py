from algos.input_optimization import InputOptimization
from algos.split_inference import SplitInference
from algos.nopeek import NoPeek
from algos.uniform_noise import UniformNoise
from algos.siamese_embedding import SiameseEmbedding
from algos.pca_embedding import PCAEmbedding
from algos.deepobfuscator import DeepObfuscator
from algos.pan import PAN
from algos.gaussian_blur import GaussianBlur
from algos.linear_correlation import LinearCorrelation

from algos.supervised_decoder import SupervisedDecoder
from algos.cloak import Cloak
from algos.shredder import Shredder
from algos.aioi import AIOI


from data.loaders import DataLoader
from models.model_zoo import Model
from utils.utils import Utils
from utils.config_utils import config_loader
from os import path


def load_config(filepath):
    return config_loader(filepath)


def load_model(config, utils):
    return Model(config["server"], utils)


def load_data(config):
    return DataLoader(config)


def load_utils(config):
    return Utils(config)


def load_algo(config, utils, dataloader=None):
    method = config["method"]
    if method == "split_inference":
        algo = SplitInference(config["client"], utils)
    elif method == "nopeek":
        algo = NoPeek(config["client"], utils)
    elif method == "uniform_noise":
        algo = UniformNoise(config["client"], utils)
    elif method == "siamese_embedding":
        algo = SiameseEmbedding(config["client"], utils)
    elif method == "pca_embedding":
        algo = PCAEmbedding(config["client"], utils)
    elif method == "deep_obfuscator":
        algo = DeepObfuscator(config["client"], utils)
    elif method == "pan":
        algo = PAN(config["client"], utils)
    elif method == "cloak":
        algo = Cloak(config["client"], utils)
    elif method == "shredder":
        algo = Shredder(config["client"], utils)
    elif method == "aioi":
        algo = AIOI(config["client"],utils)
    elif method == "gaussian_blur":
        algo = GaussianBlur(config["client"], utils)
    elif method == "linear_correlation":
        algo = LinearCorrelation(config["client"], utils)
    elif method == "supervised_decoder":
        item = next(iter(dataloader))
        z = item["z"]
        config["adversary"]["channels"] = z.shape[1]
        config["adversary"]["patch_size"] = z.shape[2]
        algo = SupervisedDecoder(config["adversary"], utils)
    elif method == "input_optimization":
        config["adversary"]["target_model_path"] = path.join(config["experiments_folder"], config["challenge_experiment"], "saved_models", "client_model.pt")
        config["adversary"]["target_model_config"] = path.join(config["experiments_folder"], config["challenge_experiment"], "configs", f"{config['adversary']['target_model']}.json")
        algo = InputOptimization(config["adversary"], utils)
    else:
        print("Unknown algorithm {}".format(config["method"]))
        exit()

    return algo
