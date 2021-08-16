import torch.utils.data as data
from torchvision import datasets
from PIL import Image
from glob import glob
from abc import abstractmethod
from copy import deepcopy
import torch
import os
import pandas as pd
import re
import data.download_pytorch_dataset as dpd


class BaseDataset(data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.format = config["format"]
        self.set_filepaths(config["path"])
        self.transforms = config["transforms"]

    def set_filepaths(self, path):
        filepaths = path + "/*.{}".format(self.format)
        self.filepaths = glob(filepaths)

    def load_image(self, filepath):
        img = Image.open(filepath)
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        filename = filepath.split('/')[-1].split('.')[0]
        img = self.load_image(filepath)
        img = self.transforms(img)
        pred_label = self.load_label(filepath, "pred")
        pred_label = self.to_tensor(pred_label)
        if self.protected_attribute == "data":
            privacy_label = img
        else:
            privacy_label = self.load_label(filepath, "privacy")
            privacy_label = self.to_tensor(privacy_label)
        sample = {'img': img, 'prediction_label': pred_label, 'private_label': privacy_label,
            'filepath': filepath, 'filename': filename}
        return sample

    def __len__(self):
        return len(self.filepaths)


class FairFace(BaseDataset):
    """docstring for FairFace"""

    def __init__(self, config):
        config = deepcopy(config)
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        try:
            if config["train"] is True:
                label_csv = pd.read_csv(config["path"] +
                                        "fairface_label_train.csv")
                config["path"] += "/train"
            else:
                label_csv = pd.read_csv(config["path"] + "fairface_label_val.csv")
                config["path"] += "/val"
            self.label_csv = label_csv.set_index("file")
        except:
            self.label_csv = None
        super(FairFace, self).__init__(config)
        self.label_mapping = {}
        self.label_mapping["race"] = {"East Asian": 0,
                                      "Indian": 1,
                                      "Black": 2,
                                      "White": 3,
                                      "Middle Eastern": 4,
                                      "Latino_Hispanic": 5,
                                      "Southeast Asian": 6}
        self.label_mapping["gender"] = {"Male": 0, "Female": 1}

    def load_label(self, filepath, label_type):
        reg_exp = r'//(.*/\d+\.{})'
        filename = re.search(reg_exp.format(self.format), filepath).group(1)
        labels_row = self.label_csv.loc[filename]
        if label_type == "pred":
            pred_label = labels_row[self.prediction_attribute]
            return self.label_mapping[self.prediction_attribute][pred_label]
        else:
            privacy_label = labels_row[self.protected_attribute]
            return self.label_mapping[self.protected_attribute][privacy_label]

class LFW(BaseDataset):
    """docstring for Labeled Faces in the Wild"""

    def __init__(self, config):
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        try:
            if config["train"] is True:
                label_csv = pd.read_csv(config["path"] +
                                        "lfw_label_train.csv")
                config["path"] += "/train"
            else:
                label_csv = pd.read_csv(config["path"] + "lfw_label_val.csv")
                config["path"] += "/val"
            self.label_csv = label_csv.set_index("file")
        except:
            self.label_csv = None
        super(LFW, self).__init__(config)
        self.label_mapping = {}
        self.label_mapping["race"] = {"Asian": 0,
                                      "White": 1,
                                      "Black": 2,
                                      "Indian": 3}
        self.label_mapping["gender"] = {"Male": 0, "Female": 1}


    def load_label(self, filepath, label_type):
        try:
            person_name = os.path.basename(filepath)[0:-len('_0000.jpg')]
            filename = person_name + '/' + os.path.basename(filepath)
            labels_row = self.label_csv.loc[filename]
            if label_type == "pred":
                pred_label = labels_row[self.prediction_attribute]
                return self.label_mapping[self.prediction_attribute][pred_label]            
            else:
                privacy_label = labels_row[self.protected_attribute]
                return self.label_mapping[self.protected_attribute][privacy_label]       
        except:
            return 1, 1

class Cifar10(BaseDataset):
    """docstring for Cifar10"""

    def __init__(self, config):
        config = deepcopy(config)
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        try:
            if config["train"] is True:
                label_csv = pd.read_csv(config["path"] +
                                        "cifar10_label_train.csv")
                config["path"] += "/train"
            else:
                label_csv = pd.read_csv(config["path"] + "cifar10_label_val.csv")
                config["path"] += "/val"
            self.label_csv = label_csv.set_index("file")
        except:
            self.label_csv = None
        super(Cifar10, self).__init__(config)
        self.label_mapping = {}
        self.label_mapping["class"] = {"airplane": 0,
                                      "automobile": 1,
                                      "bird": 2,
                                      "cat": 3,
                                      "deer": 4,
                                      "dog": 5,
                                      "frog": 6,
                                      "horse": 7,
                                      "ship": 8,
                                      "truck": 9}
        self.label_mapping["animated"] = {"no": 0,
                                      "yes": 1}

    def load_label(self, filepath, label_type):
        try:
            reg_exp = r'//(.*/\d+\.{})'
            filename = re.search(reg_exp.format(self.format), filepath).group(1)
            labels_row = self.label_csv.loc[filename]   
            if label_type == "pred":
                pred_label = labels_row[self.prediction_attribute]
                return self.label_mapping[self.prediction_attribute][pred_label]            
            else:
                privacy_label = labels_row[self.protected_attribute]
                return self.label_mapping[self.protected_attribute][privacy_label]
            # pred_label = labels_row[self.prediction_attribute]
            # privacy_label = labels_row[self.protected_attribute]
            # return self.label_mapping[self.prediction_attribute][pred_label], self.label_mapping[self.protected_attribute][privacy_label]
        except:
            return 1, 1


class CelebA(datasets.CelebA):
    def __init__(self, config):
        config = deepcopy(config)
        data_split = "train" if config["train"] else "valid" 
        self.reconstruct_data = config["protected_attribute"] == 'data'
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        self.attr_indices = {'gender': 20, 
                             'eyeglasses': 15,
                             'necklace': 37,
                             'smiling': 31,
                             'straight_hair': 32,
                             'wavy_hair': 33,
                             'big_nose': 7,
                             'mouth_open': 21}
        if self.prediction_attribute in self.attr_indices.keys():
            target_pred = 'attr'
        else:
            raise ValueError("Prediction Attribute {} is not supported.".format(self.prediction_attribute))
        if self.protected_attribute in self.attr_indices.keys():
            target_protect = 'attr'
            target_type = [target_pred, target_protect]
        elif self.protected_attribute == 'data':
            target_type = target_pred
        else:
            raise ValueError("Protected Attribute {} is not supported.".format(self.protected_attribute))
            
        super().__init__(root=config["path"], split=data_split, 
                       target_type=target_type, transform=config["transforms"],
                       download=False)
        
    def __getitem__(self, index):
        if self.reconstruct_data:
            img, pred_label = super().__getitem__(index)
            privacy_label = img
        else:
            img, (pred_label, privacy_label) = super().__getitem__(index)
        if self.prediction_attribute in self.attr_indices.keys():
            attr_index = self.attr_indices[self.prediction_attribute]
            pred_label = 1 if pred_label[attr_index] > 0 else 0
        if self.protected_attribute in self.attr_indices.keys():
            attr_index = self.attr_indices[self.protected_attribute]
            privacy_label = 1 if privacy_label[attr_index] > 0 else 0

        filename = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index])
        filename = filename.split('/')[-1].split('.')[0]
        sample = {'img': img, 'prediction_label': pred_label, 'private_label': privacy_label,
                  'filename': filename}
        return sample

class BaseDataset2(data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset2, self).__init__()
        self.format = config["format"]
        self.set_indicies(config["path"])
        self.transforms = config["transforms"]
        self.train_dict, self.val_dict = dpd.load_cifar_as_dict(config["path"])
        self.config = config
        if config["train"] is True:
            self.data_to_run_on = self.train_dict
        else:
            self.data_to_run_on = self.val_dict

    def set_indicies(self, path):
        filepaths = path + "/*.{}".format(self.format)
        num_of_images = self.data_to_run_on['set'].data.shape[0]
        self.indicies = [i for i in range(num_of_images)]

    def load_image(self, i):
        img = Image.fromarray(self.data_to_run_on['set'].data[i])
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.indicies[index]
        if self.config["train"] is True:
            filename = "train/"+str(filepath)+".jpg"
        else:
            filename = "val/"+str(filepath)+".jpg"
        img = self.load_image(filepath)
        img = self.transforms(img)
        pred_label = self.load_label(filepath, "pred")
        pred_label = self.to_tensor(pred_label)
        if self.protected_attribute == "data":
            privacy_label = img
        else:
            privacy_label = self.load_label(filepath, "privacy")
            privacy_label = self.to_tensor(privacy_label)
        sample = {'img': img, 'prediction_label': pred_label, 'private_label': privacy_label,
            'filepath': filepath, 'filename': filename}
        return sample

    def __len__(self):
        return len(self.indicies)

class Cifar10_2(BaseDataset2):
    """docstring for Cifar10"""

    def __init__(self, config):
        config = deepcopy(config)
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        self.train_dict, self.val_dict = dpd.load_cifar_as_dict(config["path"])
        self.data_to_run_on = None
        if config["train"] is True:
            self.data_to_run_on = self.train_dict
        else:
            self.data_to_run_on = self.val_dict
        super(Cifar10_2, self).__init__(config)
        self.label_mapping = {}
        self.label_mapping["class"] = {"airplane": 0,
                                      "automobile": 1,
                                      "bird": 2,
                                      "cat": 3,
                                      "deer": 4,
                                      "dog": 5,
                                      "frog": 6,
                                      "horse": 7,
                                      "ship": 8,
                                      "truck": 9}
        self.label_mapping["animated"] = {"no": 0,
                                      "yes": 1}

    def load_label(self, filepath, label_type):
        # print("filepath:", filepath)
        # print("label_type:", label_type)
        # print("self:", self.data_to_run_on)
        # print("d:", d)
        # print("-----")
        try:
            if label_type == "pred":
                label_name = self.prediction_attribute
                attr = self.prediction_attribute
            else:
                label_name = self.protected_attribute
                attr = self.protected_attribute
            d = self.data_to_run_on[label_name][filepath]
            d = self.label_mapping[attr][d]
            return d
        except:
            return 1, 1

class BaseDataset3(data.Dataset):
    """docstring for BaseDataset"""

    def __init__(self, config):
        super(BaseDataset3, self).__init__()
        self.format = config["format"]
        self.set_indicies(config["path"])
        self.transforms = config["transforms"]
        self.train_dict, self.val_dict = dpd.load_cifar_as_dict(config["path"])
        self.config = config
        if config["train"] is True:
            self.data_to_run_on = self.train_dict
        else:
            self.data_to_run_on = self.val_dict

    def set_indicies(self, path):
        filepaths = path + "/*.{}".format(self.format)
        num_of_images = self.data_to_run_on['set'].data.shape[0]
        self.indicies = [i for i in range(num_of_images)]

    def load_image(self, i):
        img = Image.fromarray(self.data_to_run_on['set'].data[i])
        return img

    @staticmethod
    def to_tensor(obj):
        return torch.tensor(obj)

    @abstractmethod
    def load_label(self):
        pass

    def __getitem__(self, index):
        filepath = self.indicies[index]
        filename = "challenge/"+str(filepath)+".jpg"
        img = self.load_image(filepath)
        img = self.transforms(img)
        pred_label = self.load_label(filepath, "pred")
        pred_label = self.to_tensor(pred_label)
        if self.protected_attribute == "data":
            privacy_label = img
        else:
            privacy_label = self.load_label(filepath, "privacy")
            privacy_label = self.to_tensor(privacy_label)
        sample = {'img': img, 'prediction_label': pred_label, 'private_label': privacy_label,
            'filepath': filepath, 'filename': filename}
        return sample

    def __len__(self):
        return len(self.indicies)

class Cifar10_3(BaseDataset3):
    """docstring for Cifar10, challenge loader"""

    def __init__(self, config):
        # 'train only with test loader'
        config = deepcopy(config)
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        self.train_dict, self.val_dict = dpd.load_cifar_as_dict(config["path"])
        self.data_to_run_on = None
        if config["train"] is True:
            self.data_to_run_on = self.train_dict
        else:
            self.data_to_run_on = self.val_dict
        super(Cifar10_3, self).__init__(config)
        self.label_mapping = {}
        self.label_mapping["class"] = {"airplane": 0,
                                      "automobile": 1,
                                      "bird": 2,
                                      "cat": 3,
                                      "deer": 4,
                                      "dog": 5,
                                      "frog": 6,
                                      "horse": 7,
                                      "ship": 8,
                                      "truck": 9}
        self.label_mapping["animated"] = {"no": 0,
                                      "yes": 1}

    def load_label(self, filepath, label_type):
        try:
            if label_type == "pred":
                label_name = self.prediction_attribute
                attr = self.prediction_attribute
            else:
                label_name = self.protected_attribute
                attr = self.protected_attribute
            d = self.data_to_run_on[label_name][filepath]
            d = self.label_mapping[attr][d]
            return d
        except:
            return 1, 1

def load_challenge_data_set(experiment_path):
    challenge_dir = os.path.join(experiment_path, "challenge")
    log_dir = os.path.join(experiment_path, "logs")
    pts = {int(i.split(".")[0]): str(i) for i in sorted(os.listdir(challenge_dir), key=lambda s: int(s.split(".")[0]))}
    return pts
