import torchvision
from PIL import Image
import pandas as pd
import os


def save_as_csv():
    pass

def load_CIFAR10_dataset(output_dir):
    try:
        trainset = torchvision.datasets.CIFAR10(root=output_dir, train=True)
        valset = torchvision.datasets.CIFAR10(root=output_dir, train=False)
    except:
        trainset = torchvision.datasets.CIFAR10(root=output_dir, train=True, download=True)
        valset = torchvision.datasets.CIFAR10(root=output_dir, train=False, download=True)
    return trainset, valset

def translate_into_images(dataset, output_dir, prefix=None):
    if prefix:
        new_dir = os.path.join(output_dir, prefix)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    num_of_images = dataset.data.shape[0]
    filenames = []
    filepaths = []
    for i in range(num_of_images):
        filename = str(i) + ".jpg"
        if prefix:
            filename = os.path.join(prefix, filename)
        filepath = os.path.join(output_dir, filename)
        img = Image.fromarray(dataset.data[i])
        img.save(filepath)
        filenames.append(filename)
        filepaths.append(filepath)
    return filenames, filepaths

def load_labels(dataset):
    mappings = dataset.class_to_idx
    inv_map = {v: k for k, v in mappings.items()}
    new_labels = []
    for label in dataset.targets:
        new_labels.append(inv_map[label])
    return new_labels

def map_class_to_animated(dataset):
    living = set(['bird', 'cat', 'deer', 'dog', 'frog', 'horse'])
    mappings = dataset.class_to_idx
    inv_map = {v: k for k, v in mappings.items()}
    new_labels = []
    for label in dataset.targets:
        l = inv_map[label]
        if l in living:
            new_labels.append("yes")
        else:
            new_labels.append("no")
    return new_labels

def create_csv(dic, output_dir, filename):
    df = pd.DataFrame(dic)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

def save_and_organize_cifar10_dataset(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    trainset, valset = load_CIFAR10_dataset(output_dir)

    train_files, train_filepaths = translate_into_images(trainset, output_dir, "train")
    val_files, val_filepaths = translate_into_images(valset, output_dir, "val")
    train_labels = load_labels(trainset)
    val_labels = load_labels(valset)

    train_dict = dict()
    train_dict["file"] = train_files
    train_dict["animated"] = map_class_to_animated(trainset)
    train_dict["class"] = train_labels
    create_csv(train_dict, output_dir, "cifar10_label_train.csv")

    val_dict = dict()
    val_dict["file"] = val_files
    val_dict["animated"] = map_class_to_animated(valset)
    val_dict["class"] = val_labels
    create_csv(val_dict, output_dir, "cifar10_label_val.csv")

def load_cifar_as_dict(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    trainset, valset = load_CIFAR10_dataset(output_dir)

    train_files, train_filepaths = translate_into_images(trainset, output_dir, "train")
    val_files, val_filepaths = translate_into_images(valset, output_dir, "val")
    train_labels = load_labels(trainset)
    val_labels = load_labels(valset)

    train_dict = dict()
    train_dict["set"] = trainset
    train_dict["file"] = train_files
    train_dict["animated"] = map_class_to_animated(trainset)
    train_dict["class"] = train_labels

    val_dict = dict()
    val_dict["set"] = valset
    val_dict["file"] = val_files
    val_dict["animated"] = map_class_to_animated(valset)
    val_dict["class"] = val_labels
    
    return train_dict, val_dict


if __name__ == '__main__':
    base_dir = "/home/mit6_91621/cybop/"
    output_dir = base_dir + "cifar10"
    save_and_organize_cifar10_dataset(output_dir)
