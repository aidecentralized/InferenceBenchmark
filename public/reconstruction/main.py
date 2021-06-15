import argparse
from pytorch_msssim import SSIM, MS_SSIM
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter

from adv_pred_model import AdversaryModelPred
from AEs import Autoencoder, MinimalDecoder
from dataset_utils import ImageTensorFolder, TensorPredictionData
import os
import numpy as np
from shutil import copytree, copy2
from glob import glob
# from generate_ir import get_client_model

random_seed = 100
torch.manual_seed(random_seed)
np.random.seed(random_seed)

FAIRFACE_SPLIT_LAYER_INPUT_NC = {1: 64, 2: 64, 3: 64, 4: 64, 5: 128, 6:256, 7:512}

def apply_transform(batch_size, train_split, goal_data_dir, tensor_data_dir,
                    img_fmt, tns_fmt, task):
    if task == 'gender':
        dataset = TensorPredictionData(tensor_data_dir, goal_data_dir, 
                                       pred_gender=True, tns_fmt=tns_fmt)
    elif task == 'smile':
        dataset = TensorPredictionData(tensor_data_dir, goal_data_dir, 
                                       pred_smile=True, tns_fmt=tns_fmt)
    elif task == 'race':
        dataset = TensorPredictionData(tensor_data_dir, goal_data_dir, 
                                       pred_race=True, tns_fmt=tns_fmt)
    else:
        trainTransform = transforms.Compose([transforms.ToTensor(),])
        dataset = ImageTensorFolder(img_path=goal_data_dir, tensor_path=tensor_data_dir,
                                    img_fmt=img_fmt, tns_fmt=tns_fmt, transform=trainTransform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             sampler=test_sampler)


    return trainloader, testloader

def denormalize(img, dataset="imagenet"):
    """
    data is normalized with mu and sigma, this function puts it back
    """
    if dataset == "cifar10":
        c_std = [0.247, 0.243, 0.261]
        c_mean = [0.4914, 0.4822, 0.4466]
    elif dataset == "imagenet":
        c_std = [0.229, 0.224, 0.225]
        c_mean = [0.485, 0.456, 0.406]
    for i in [0, 1, 2]:
        img[i] = img[i] * c_std[i] + c_mean[i]
    return img

def save_images(input_imgs, output_imgs, epoch, path, img_nums, offset=0, batch_size=64):
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for img_idx in range(input_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, img_nums[img_idx])  # offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, img_nums[img_idx])  # offset * batch_size + img_idx)
        #inp_img = denormalize(input_imgs[img_idx])
        #out_img = denormalize(output_imgs[img_idx])
        save_image(input_imgs[img_idx], inp_img_path)
        save_image(output_imgs[img_idx], out_img_path)

def copy_source_code(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    for file_ in glob(r'./*.py'):
        copy2(file_, path)
    copytree("clients/", path + "clients/")

def main(architecture, task, goal_data_dir, tensor_data_dir, img_fmt, tns_fmt,
         loss_fn, train_split, batch_size, num_epochs, train_output_freq, 
         test_output_freq, split_layer, gpu_id):

    device = torch.device('cuda:{}'.format(gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    print("Using device as {}".format(device))

    output_path = "./output/{}".format(architecture)
    train_output_path = "{}/train".format(output_path)
    test_output_path = "{}/test".format(output_path)
    tensorboard_path = "{}/tensorboard/".format(output_path)
    source_code_path = "{}/sourcecode/".format(output_path)
    model_path = "{}/model.pt".format(output_path)

    writer = SummaryWriter(logdir=tensorboard_path)

    if task == 'gender' or task == 'smile' or task == 'race':
        decoder = AdversaryModelPred(split_layer).to(device)
        train_loss_fn = nn.CrossEntropyLoss()
    else:
        input_nc = FAIRFACE_SPLIT_LAYER_INPUT_NC[split_layer]
        decoder = Autoencoder(input_nc=input_nc, output_nc=3, split_layer=split_layer).to(device)
        #decoder = MinimalDecoder(input_nc=64, output_nc=3, input_dim=112, output_dim=224).to(device)
        torch.save(decoder.state_dict(), model_path)
        # decoder.load_state_dict(torch.load(model_path))
        # copy_source_code(source_code_path)
    
        if (loss_fn.lower() == 'mse'):
            train_loss_fn = nn.MSELoss()
        elif (loss_fn.lower() == 'ssim'):
            ssim_value = SSIM(data_range=255, size_average=True, channel=3)
            train_loss_fn = lambda x, y: 1 - ssim_value(x, y)
        elif (loss_fn.lower() == 'ms_ssim'):
            ssim_loss_fn = MS_SSIM(data_range=255, size_average=True, channel=3)
            train_loss_fn = lambda x, y: 1 - ssim_value(x, y)
        else:
            raise ValueError("Loss function {} not recognized".format(loss_fn.lower()))

    trainloader, testloader = apply_transform(batch_size, train_split, 
                                        goal_data_dir, tensor_data_dir, 
                                        img_fmt, tns_fmt, task)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    #client_model = get_client_model().to(device)
    #for param in client_model.parameters():
    #    param.requires_grad = False

    round_ = 0

    for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
        for num, data in enumerate(trainloader, 1):
            img, ir, img_nums = data
            if task != 'gender' and task != 'smile' and task != 'race':  # for gender 'img' is actually male/female label
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            #ir = client_model(img)
            output = decoder(ir)

            reconstruction_loss = train_loss_fn(output, img)
            train_loss = reconstruction_loss

            writer.add_scalar('loss/train', train_loss.item(), len(trainloader) * epoch + num)
            writer.add_scalar('loss/train_loss/reconstruction', reconstruction_loss.item(), len(trainloader) * epoch + num)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        if (epoch + 1) % train_output_freq == 0 and task != 'gender' and task != 'smile' and task != 'race':
            save_images(img, output, epoch, train_output_path, img_nums)  # offset=0, batch_size=batch_size)
        
        pred_correct = 0
        total = 0
        for num, data in enumerate(testloader, 1):
            img, ir, img_nums = data
            if task != 'gender' and task != 'smile' and task != 'race':  
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            #ir = client_model(img)
            output = decoder(ir)
            if task == 'gender' or task == 'smile' or task == 'race':
                pred_correct += (output.argmax(dim=1) == img).sum().item()
            total += int(ir.shape[0])

            reconstruction_loss = train_loss_fn(output, img)
            test_loss = reconstruction_loss

            writer.add_scalar('loss/test', test_loss.item(), len(testloader) * epoch + num)
            writer.add_scalar('loss/test_loss/reconstruction', reconstruction_loss.item(), len(testloader) * epoch + num)
        
        pred_acc = pred_correct / total
        writer.add_scalar("test/pred_accuracy", pred_acc, len(testloader) * epoch)

        if (epoch + 1) % test_output_freq == 0 and task != 'gender' and task != 'smile' and task != 'race':
            for num, data in enumerate(testloader):
                img, ir, img_nums = data
                if task != 'gender' and task != 'smile' and task != 'race':
                    img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)

                #ir = client_model(img)
                output_imgs = decoder(ir)

                save_images(img, output_imgs, epoch, test_output_path, img_nums)  # offset=num, batch_size=batch_size)

        for name, param in decoder.named_parameters():
            writer.add_histogram("params/{}".format(name), param.clone().cpu().data.numpy(), epoch)

        model_path = "{}/model_{}.pt".format(output_path, epoch)
        torch.save(decoder.state_dict(), model_path)
        print("epoch [{}/{}], train_loss {:.4f}, test_loss {:.4f}, pred_acc {:.4f}".format(epoch + 1,
                                        num_epochs, train_loss.item(), test_loss.item(), pred_acc))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='pruning-network-fairface-reconstruction-split6-ratio0.3-1')
    parser.add_argument('--split_layer', type=int, default=1)
    parser.add_argument('--tensor_data_dir', type=str, default='/home/emizhang/chap/experiments/pruning_network_fairface_resnet18_scratch_split5_ratio0.9_data_2/challenge',
                        help='intermediate image data directory')
    parser.add_argument('--goal_data_dir', type=str, default='/mas/camera/Datasets/Faces/fairface/val/',
                        help='training image data directory or data labels')
    parser.add_argument('--task', type=str, default="data", help='choose between data, and gender (for celeba)')
    
    parser.add_argument('--img_fmt', type=str, default='jpg', help='format of training images, one of png, jpg, jpeg, npy, pt')
    parser.add_argument('--tns_fmt', type=str, default='pt', help='format of tensor data, one of png, jpg, jpeg, npy, pt')
    parser.add_argument('--train_split', type=float, default=0.9, help='ratio of data to use for training')
    parser.add_argument('--loss_fn', type=str, default='mse', help='loss function to use for training, one of mse, ssim, ms_ssim')

    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    # parser.add_argument('--optimizer_pick', type=str, default="Adam", help='choose optimizer between Adam and SGD')
    parser.add_argument('--num_epochs', type=int, default=500, help='maximum number of epochs')
    parser.add_argument("--train_output_freq", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--test_output_freq", type=int, default=50, help="interval between saving model weights")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id to use if training on cuda")

    opt = parser.parse_args()
    main(architecture=opt.architecture,
         task=opt.task,
         goal_data_dir=opt.goal_data_dir,
         tensor_data_dir=opt.tensor_data_dir,
         img_fmt=opt.img_fmt,
         tns_fmt=opt.tns_fmt,
         loss_fn=opt.loss_fn,
         train_split=opt.train_split,
         batch_size=opt.batch_size, 
         num_epochs=opt.num_epochs, 
         train_output_freq=opt.train_output_freq, 
         test_output_freq=opt.test_output_freq,
         split_layer=opt.split_layer,
         gpu_id=opt.gpu_id)
