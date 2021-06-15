import torch.utils.data as data
import torch
import pandas as pd
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import numpy as np

class ImageTensorFolder(data.Dataset):

    def __init__(self, img_path, tensor_path, img_fmt="npy", tns_fmt="npy", transform=None):
        self.img_fmt = img_fmt
        self.tns_fmt = tns_fmt
        # self.img_paths = self.get_all_files(img_path, file_format=img_fmt)
        self.tensor_paths = self.get_all_files(tensor_path, file_format=tns_fmt)
        self.get_img_files_from_tensors(img_path, tensor_path)

        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def get_all_files(self, path, file_format="png"):
        filepaths = path + "/*.{}".format(file_format)
        files = glob(filepaths)
        print(files[0:10])
        return files

    def get_img_files_from_tensors(self, img_path, tensor_path):
        """
        Only get image files corresponding to tensors for training
        """
        self.img_paths = []
        for tensor in self.tensor_paths:
            img_name = tensor.replace(tensor_path + '/', '').replace('_rec', '').replace(self.tns_fmt, self.img_fmt)
            self.img_paths.append(img_path + img_name)

    def load_img(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            img = Image.open(filepath)
            img = img.resize((224, 224))  # TODO remove this -- hardcoded for celeba
            # Drop alpha channel
            if self.to_tensor(img).shape[0] == 4:
                img = self.to_tensor(img)[:3, :, :]
                img = self.to_pil(img)
        elif file_format == "npy":
            img = np.load(filepath)
            #cifar10_mean = [0.4914, 0.4822, 0.4466]
            #cifar10_std = [0.247, 0.243, 0.261]
            img = np.uint8(255 * img)
            img = self.to_pil(img)
        elif file_format == "pt":
            img = torch.load(filepath)
        else:
            print("Unknown format")
            exit()
        return img

    def load_tensor(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            tensor = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(tensor).shape[0] == 4:
                tensor = self.to_tensor(tensor)[:3, :, :]
            else:
                tensor = self.to_tensor(tensor)
        elif file_format == "npy":
            tensor = np.load(filepath)
            tensor = self.to_tensor(tensor)
        elif file_format == "pt":
            tensor = torch.load(filepath)
            tensor.requires_grad = False
        return tensor

    def __getitem__(self, index):
        img = self.load_img(self.img_paths[index], file_format=self.img_fmt)
        img_num = self.img_paths[index].split("/")[-1].replace('_rec', '').split(".")[0]
        intermed_rep = self.load_tensor(self.tensor_paths[index], file_format=self.tns_fmt)
        if self.transform is not None:
            img = self.transform(img)
        return img, intermed_rep, img_num

    def __len__(self):
        return len(self.img_paths)


class TensorPredictionData(data.Dataset):
    def __init__(self, tensor_path, labels_path, pred_gender=False, 
                 pred_smile=False, pred_race=False, tns_fmt="pt"):
        self.tensor_paths = self.get_all_files(tensor_path, file_format=tns_fmt)
        self.tns_fmt = tns_fmt
        self.pred_gender = pred_gender
        self.pred_smile = pred_smile
        self.pred_race = pred_race
        self.gender_index = 20
        self.smile_index = 31

        if self.pred_gender or self.pred_smile:
            self.label_dict = get_celeba_attr_dict(labels_path)
        elif self.pred_race:
            label_csv = pd.read_csv(labels_path)
            self.label_csv = label_csv.set_index("file")
            self.label_mapping = {}
            self.label_mapping["race"] = {"East Asian": 0,
                                          "Indian": 1,
                                          "Black": 2,
                                          "White": 3,
                                          "Middle Eastern": 4,
                                          "Latino_Hispanic": 5,
                                          "Southeast Asian": 6}
    
    def get_all_files(self, path, file_format):
        filepaths = path + "/*.{}".format(file_format)
        files = glob(filepaths)
        return files

    def load_tensor(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            tensor = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(tensor).shape[0] == 4:
                tensor = self.to_tensor(tensor)[:3, :, :]
            else:
                tensor = self.to_tensor(tensor)
        elif file_format == "npy":
            tensor = np.load(filepath)
            tensor = self.to_tensor(tensor)
        elif file_format == "pt":
            tensor = torch.load(filepath)
            tensor.requires_grad = False
        return tensor

    def __getitem__(self, index):
        img_num = self.tensor_paths[index].split("/")[-1].split(".")[0]
        intermed_rep = self.load_tensor(self.tensor_paths[index], file_format=self.tns_fmt)
        if self.pred_gender:
            label = int(self.label_dict[img_num][self.gender_index])
            label = 1 if label > 0 else 0
        elif self.pred_smile:
            label = int(self.label_dict[img_num][self.smile_index])
            label = 1 if label > 0 else 0
        elif self.pred_race:
            filename = 'train/{}.jpg'.format(index+1)
            labels_row = self.label_csv.loc[filename]
            label = self.label_mapping["race"][labels_row['race']]
        else:
            raise ValueError("only gender prediction supported for now")

        return label, intermed_rep, img_num

    def __len__(self):
        return len(self.tensor_paths)


#returns dict of image_num to list of attributes
def get_celeba_attr_dict(attr_path):
    rfile = open(attr_path, 'r' ) 
    texts = rfile.read().split("\n") 
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    label_dict = {}
    for txt in texts[2:]:
        if txt == '': continue
        row = np.array(txt.split(" "))
        row = row[row!= ""]
        img_num = row[0].split('.')[0]
        label_dict[img_num] = row[1:]

    return label_dict
