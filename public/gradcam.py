import argparse
from torch.autograd import Variable
import test_all_experiment_models as t
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torchvision import models
from adv_models import AdversaryModelGen, AdversaryModelPred, reconstruction_loss
import combine_models_for_analysis as comb
from torchvision import datasets, transforms
from PIL import Image
import os
import random

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, prune=None, pruner=None):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        self.target_layers = target_layers
        self.prune = prune
        self.pruner = pruner

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        pruned = False
        for name, module in self.model._modules.items():
            if name != '2':
                for n_name, n_module in module.model._modules.items():
                    if n_module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name.lower():
                        x = n_module(x)
                        x = x.view(x.size(0),-1)
                    else:
                        x = n_module(x)
                if self.prune and not pruned:
                    x = self.pruner(x)[0]
                    pruned = True
            else:
                x = module(x)
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def chap_preprocess_image(img_path):
    device = t.load_device()
    trainTransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
    im1 = Image.open(img_path)
    n_img = trainTransform(im1)
    inputloader = torch.utils.data.DataLoader(
            [n_img], batch_size=1, num_workers=5)
    for batch_idx, sample in enumerate(inputloader):
        input = sample
    input = input.requires_grad_(True).to(device)
    input.retain_grad()
    return input


def show_cam_on_image(img, mask, name=None, output_dir = None, output_prefix = None, pruned = None):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    if output_dir and name:
        n_heat = output_dir + output_prefix + "heat_"+ name
        print("heat map:", n_heat)
        cv2.imwrite(n_heat, np.uint8(255 * cam))
        return n_heat


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, prune=None, pruner=None):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if torch.cuda.is_available():
            self.cuda = True
        if self.cuda:
            self.model = model.cuda()
        self.prune = prune
        self.pruner = pruner

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, prune, pruner)

    def forward(self, input):
        ni = input
        pruned = False
        for name, module in self.model._modules.items():
            if name != '2':
                for n_name, n_module in module.model._modules.items():
                    ni = n_module(ni)
                if self.prune and not pruned:
                    ni = pruner(ni)[0]
                    pruned = True
            else:
                ni = module(ni)
        return ni

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, index


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda, prune=None, pruner=None):
        self.model = model
        self.model.eval()
        self.prune = prune
        self.pruner = pruner
        self.cuda = use_cuda
        if torch.cuda.is_available():
            self.cuda = True
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        ni = input
        pruned = False
        for name, module in self.model._modules.items():
            if name != '2':
                for n_name, n_module in module.model._modules.items():
                    ni = n_module(ni)
                if self.prune and not pruned:
                    ni = self.pruner(ni)[0]
                    pruned = True
            else:
                ni = module(ni) 
        print("output:", ni)
        return ni

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('-f', help='ehhhh')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def load_client_server_models(experiment_path):
    s, c, p, a = comb.load_em(experiment_path)
    split_layer = int(comb.get_split_layer(experiment_path))
    s._modules['model'] = s.model[split_layer + 1:]
    c._modules['model'] = c.model[0:split_layer + 1]
    model = torch.nn.Sequential(c, s, nn.Softmax(dim=1))  # interested in model[1].model[7] for clinet / server combo
    # model = torch.nn.Sequential(c, s)

    feature_module = None # Meant to be last module before avg layer
    for name, module in model._modules.items():
        if name != '2':
            for n_name, n_module in module.model._modules.items():
                if n_name == "7": # Layer 7 is right before the avg / pooling layers
                    feature_module = n_module
    if p:
        p.eval()
    return model, p, feature_module

def load_client_adversary_models(experiment_path):
    s, c, p, a = comb.load_em(experiment_path)
    split_layer = int(comb.get_split_layer(experiment_path))
    a._modules['model'] = a.model[split_layer + 1:]
    c._modules['model'] = c.model[0:split_layer + 1]
    model = torch.nn.Sequential(c, a, nn.Softmax(dim=1))  # interested in model[1].model[7] for clinet / server combo
    # model = torch.nn.Sequential(c, a)

    feature_module = None # Meant to be last module before avg layer
    for name, module in model._modules.items():
        if name != '2':
            for n_name, n_module in module.model._modules.items():
                if n_name == "7":
                    feature_module = n_module
    if p:
        p.eval()
    return model, p, feature_module


def run_exp(model, feature_module, im_path, output_dir=None, output_prefix=None, pruner=None, target_index=None):
    if not output_prefix:
        output_prefix = ''
    if not pruner:
        prune = False
        grad_cam = GradCam(model=model, feature_module=feature_module, \
                       target_layer_names=["0", "1"], use_cuda=args.use_cuda)  # target layers represent each layer of interest within the feature module
    else:
        output_prefix = output_prefix + 'pruned_'
        prune = True
        grad_cam = GradCam(model=model, feature_module=feature_module, \
                       target_layer_names=["0", "1"], use_cuda=args.use_cuda, \
                       prune = prune, pruner = pruner)  # target layers represent each layer of interest within the feature module
    name = im_path.split("/")[-1]
    print("path:", im_path)
    img = cv2.imread(im_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    input = chap_preprocess_image(im_path)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    mask = grad_cam(input, target_index)
    n_heat = None
    if output_dir:
        n_heat = show_cam_on_image(img, mask, name, output_dir, output_prefix, prune)

    if not pruner:
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    else:
        prune = True
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda, \
                                        prune = prune, pruner = pruner)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    n_gb = None
    n_cam = None
    if output_dir:
        n_gb = output_dir + output_prefix + 'gb_' + name, gb
        n_cam = output_dir + output_prefix + 'cam_' + name, cam_gb
        cv2.imwrite(n_gb)
        cv2.imwrite(n_cam)
    return n_gb, n_cam, n_heat

def gen_heat_map(model, feature_module, im_path, output_dir=None, output_prefix=None, pruner=None, target_index=None):
    args = get_args()
    if not output_prefix:
        output_prefix = ''
    if not pruner:
        prune = False
        grad_cam = GradCam(model=model, feature_module=feature_module, \
                       target_layer_names=["0", "1"], use_cuda=args.use_cuda)  # target layers represent each layer of interest within the feature module
    else:
        output_prefix = output_prefix + 'pruned_'
        prune = True
        grad_cam = GradCam(model=model, feature_module=feature_module, \
                       target_layer_names=["0", "1"], use_cuda=args.use_cuda, \
                       prune = prune, pruner = pruner)  # target layers represent each layer of interest within the feature module
    name = im_path.split("/")[-1]
    print("path:", im_path)
    img = cv2.imread(im_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    # input = preprocess_image(img)
    input = chap_preprocess_image(im_path)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    mask, target_index = grad_cam(input, target_index)
    n_heat = None
    if output_dir:
        n_heat = show_cam_on_image(img, mask, name, output_dir, output_prefix, prune)

    return n_heat, target_index

def load_image_path_from_dir(path):
    imgs = []
    files = os.listdir(path)
    for f in files:
        if f.endswith('.jpg'):
            imgs.append(f)
    return imgs

def load_random_img(imgs):
    img = random.choice(imgs)
    return img

def transpose(imgs):
    rows = len(imgs)
    cols = len(imgs[0])
    new_imgs = []
    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(imgs[r][c])
        new_imgs.append(new_row)
    return new_imgs


def gen_heat_imgs(experiments, imgs_path=None, imgs=None, val_output_dir=None):
    if not experiments:
        print("need to give experiment(s) path in a list to load")
        return
    if not val_output_dir:
        val_output_dir = 'v_output/'
    if not imgs_path:
        imgs_path = comb.get_config(experiments[0]).get("dataset_path") + 'val/'
    if not imgs:
        imgs = load_image_path_from_dir(imgs_path)
        imgs = [os.path.join(imgs_path, load_random_img(imgs))]

    all_imgs = []
    all_indicies = []
    for experiment_path in experiments:
        # print("experiment:", experiment_path)
        row_of_imgs = []
        row_of_indicies = []
        for im_path in imgs:
            # print("im_path:", im_path)
            output_dir = experiment_path + val_output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model, pruner, feature_module = load_client_server_models(experiment_path)
            if "nopruning" in experiment_path:
                pruner=None
            n_heat, target = gen_heat_map(model, feature_module, im_path, output_dir=output_dir, output_prefix="c_p_", pruner=pruner)
            row_of_imgs.append(n_heat)
            row_of_indicies.append(target)
            model, pruner, feature_module = load_client_adversary_models(experiment_path)
            if "nopruning" in experiment_path:
                pruner=None
            n_heat, target = gen_heat_map(model, feature_module, im_path, output_dir=output_dir, output_prefix="a_p_", pruner=pruner)
            row_of_imgs.append(n_heat)
            row_of_indicies.append(target)
        all_imgs.append(row_of_imgs)
        all_indicies.append(row_of_indicies)
    all_imgs = transpose(all_imgs)
    all_indicies = transpose(all_indicies)
    return all_imgs, all_indicies

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)

    im_path = '/Users/ethangarza/pytorch-grad-cam/examples/test_image_set/1.jpg'
    im_path2 = '/Users/ethangarza/pytorch-grad-cam/examples/test_image_set/9999.jpg'

    im_paths = [im_path, im_path2]

    img_train_direcotry = '/Users/ethangarza/FairFace/fairface-img-margin025-trainval/train/' # '/mas/camera/Datasets/Faces/fairface/train/'
    img_val_directory = '/Users/ethangarza/FairFace/fairface-img-margin025-trainval/val/' # '/mas/camera/Datasets/Faces/fairface/val/'
    im_train_paths = load_image_path_from_dir(img_train_direcotry)
    im_val_paths = load_image_path_from_dir(img_val_directory)

    train_output_dir = 't_output/'
    val_output_dir = 'v_output/'
    experiment_path1 = '/Users/ethangarza/experimentswarmup_30_pruning_network_fairface_resnet18_scratch_split7_ratio0.6_race_2/'
    experiment_path2 = '/Users/ethangarza/chap/chap/experiments/pruning_nopruning_fairface_resnet18_scratch_split6_ratio0.3_1'


    experiment_paths = [experiment_path1, experiment_path2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgs = []
    for experiment_path in experiment_paths:
        row_of_imgs = []
        for im_path in im_val_paths:
            output_dir = experiment_path + val_output_dir
            im_path = img_val_directory + im_path
            model, pruner, feature_module = load_client_server_models(experiment_path)
            n_heat = gen_heat_map(model, feature_module, im_path, output_dir=output_dir, output_prefix="c_p_", pruner=pruner)
            row_of_imgs.append(n_heat)
            model, pruner, feature_module = load_client_adversary_models(experiment_path)
            n_heat = gen_heat_map(model, feature_module, im_path, output_dir=output_dir, output_prefix="a_p_", pruner=pruner)
            row_of_imgs.append(n_heat)
        imgs.append(row_of_imgs)
