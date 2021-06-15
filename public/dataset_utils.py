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
import download_pytorch_dataset as dpd
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def infer_data_info(data_dir, name, protected_attribute, prediction_attribute):
    base = os.path.join(data_dir, name)
    train_labels_path = os.path.join(base, name+"_label_train.csv")
    csv_df = pd.read_csv(train_labels_path)
    csv_dict, class_dict = infer_csv_values(csv_df)
    server_logits = len(class_dict[prediction_attribute].values())
    adv_logits = len(class_dict[protected_attribute].values())
    return server_logits, adv_logits


def infer_csv_values(csv_df):
    columns = list(csv_df.columns)
    csv_dic = csv_df.to_dict('dict')
    class_dict = dict()
    for key in columns:
        if str(key) == 'file' or str(key) == 'files': 
            continue
        class_mappings = dict()
        i = 0
        for val in list(set(csv_dic.get(key, {}).values())):
            class_mappings[val] = i
            i += 1
        class_dict[key] = class_mappings
    return csv_dic, class_dict


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


class Custom(BaseDataset):
    """docstring for any Dataset that follows the rules of engagement"""

    def __init__(self, config, name):
        config = deepcopy(config)
        self.prediction_attribute = config["prediction_attribute"]
        self.protected_attribute = config["protected_attribute"]
        base = os.path.join(config["path"], name)
        try:
            if config["train"] is True:
                train_labels_path = os.path.join(base, name + "_label_train.csv")
                label_csv = pd.read_csv(train_labels_path)
                config["path"] = os.path.join(base, "train")
            else:
                val_labels_path = os.path.join(base, name + "_label_val.csv")
                label_csv = pd.read_csv(val_labels_path)
                config["path"] = os.path.join(base, "val")
            self.csv_dict, self.label_mapping = infer_csv_values(label_csv)
            self.label_csv = label_csv.set_index("file")
        except:
            self.label_csv = None
            self.csv_dict = None
            self.label_mapping = config.get('mappings', {})
        super(Custom, self).__init__(config)

    def get_mappings(self):
        return self.label_mapping

    def load_label(self, filepath, label_type):
        try:
            reg_exp = r'//(.*/\d+\.{})'
            filepath = os.path.normpath(filepath)
            base_path, file = os.path.split(filepath)
            base_path, dir_of_filename = os.path.split(base_path)
            filename = str(os.path.join(dir_of_filename, file))
            labels_row = self.label_csv.loc[filename]
            if label_type == "pred":
                pred_label = labels_row[self.prediction_attribute]
                return self.label_mapping[self.prediction_attribute][pred_label]
            else:
                privacy_label = labels_row[self.protected_attribute]
                return self.label_mapping[self.protected_attribute][privacy_label]
        except:
            return 0, 0
