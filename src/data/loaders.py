import numpy as np
import torch
from torchvision import transforms
from data.dataset_utils import FairFace, CelebA, Cifar10, LFW#, UTKFace, Cifar10_2
from data.dataset_utils import Challenge

class DataLoader():
    def __init__(self, config):
        self.config = config
        self.train, self.test = self.setup_data_pipeline()

    def get_split(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.config["train_split"] * dataset_size))
        np.random.shuffle(indices)

        train_indices, test_indices = indices[:split], indices[split:]
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        return train_dataset, test_dataset

    def setup_data_pipeline(self):
        self.IM_SIZE = self.config["img_size"]
        trainTransform = transforms.Compose([
            transforms.Resize((self.IM_SIZE, self.IM_SIZE)),
            transforms.ToTensor()])

        isattack = self.config["experiment_type"] == "attack"
        if isattack:
            config = {
                "transforms": trainTransform,
                "train": False,
                "val": False,
                "challenge": True,
                "path": self.config["dataset_path"],
                "prediction_attribute": "data",
                "protected_attribute": self.config["protected_attribute"],
                "challenge_dir": self.config["challenge_dir"],
                "dataset": self.config["dataset"]
            }
            # Attack mode only uses test dataset. We will split it below
            train_config, test_config = config, config
        else:
            train_config = {"transforms": trainTransform,
                            "train": True,
                            "val": False,
                            "challenge": False,
                            "path": self.config["dataset_path"],
                            "prediction_attribute": self.config["prediction_attribute"],
                            "protected_attribute": self.config["protected_attribute"]}
            test_config = {"transforms": trainTransform,
                        "train": False,
                        "val": True,
                        "challenge": False,
                        "path": self.config["dataset_path"],
                        "prediction_attribute": self.config["prediction_attribute"],
                        "protected_attribute": self.config["protected_attribute"]}

        if self.config["dataset"] == "fairface":
            train_config["format"] = "jpg"
            test_config["format"] = "jpg"
            train_dataset = FairFace(train_config)
            test_dataset = FairFace(test_config)
        elif self.config["dataset"] == "celeba":
            train_dataset = CelebA(train_config)
            test_dataset = CelebA(test_config)
        elif self.config["dataset"] == "cifar10":
            train_config["format"] = "jpg"
            test_config["format"] = "jpg"
            train_dataset = Cifar10(train_config)
            test_dataset = Cifar10(test_config)
        elif self.config["dataset"] == "lfw":
            train_config["format"] = "jpg"
            test_config["format"] = "jpg"
            train_dataset = LFW(train_config)
            test_dataset = LFW(test_config)
        elif self.config["dataset"] == "UTKFace":
            train_config["format"] = "jpg"
            dataset = UTKFace(train_config)

        if self.config["split"] is True and not isattack:
            train_dataset, test_dataset = self.get_split(dataset)


        if isattack:
            dataset = Challenge(train_config)
            train_dataset, test_dataset = self.get_split(dataset)

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config["train_batch_size"],
            shuffle=True, num_workers=5, drop_last=True)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config["test_batch_size"],
            shuffle=False, num_workers=5, drop_last=True)

        return self.trainloader, self.testloader
