import torch
import torch.nn as nn
from functools import reduce
from operator import __add__

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    
def conv_conv_pool(input_,
                   n_filters,
                   pool=True,
                   activation=nn.LeakyReLU(),
                   use_bn=True):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activation functions
        use_bn: True/False use batch_norm or instance_norm
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    kernel_sizes=(3,3)
    modules = []
    for i, F in enumerate(n_filters):
        in_channel = input_ if i == 0 else F
        conv_padding = reduce(__add__, 
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
        pad = nn.ZeroPad2d(conv_padding)
        net = nn.Conv2d(
            in_channel,
            F,
            kernel_sizes
        )
        bn = nn.BatchNorm2d(F)
        modules += [pad, net, bn, activation]
        
    return nn.Sequential(*modules)

def upconv_2D_MB(n_filter, name):
    """Up SAMPLING `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

def upconv_concat_MB(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D_MB(n_filter, name)(inputA)
    return torch.cat(
        (up_conv, input_B), axis=1)

class StochasticUNet(nn.Module):
    def __init__(self):
        super().__init__()
        use_bn=True
        self.conv1 = conv_conv_pool(3, [16,16], use_bn=use_bn)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = conv_conv_pool(16, [32,32], use_bn=use_bn)
        self.pool2 = nn.MaxPool2d((2,2))
        self.conv3 = conv_conv_pool(32, [64,64],  use_bn=use_bn)
        self.pool3 = nn.MaxPool2d((2,2))
        self.conv4 = conv_conv_pool(64, [128,128],  use_bn=use_bn)
        self.pool4 = nn.MaxPool2d((2,2))
        self.conv5 = conv_conv_pool(128, [256,256], pool=False, use_bn=use_bn)
        
        self.up6 = upconv_2D_MB(128, 6)
        self.conv6 = conv_conv_pool(640, [128,128], pool=False, use_bn=use_bn)
        
        self.up7 = upconv_2D_MB(64, 7)
        self.conv7 = conv_conv_pool(192, [64,64], pool=False, use_bn=use_bn)
        
        self.up8 = upconv_2D_MB(32, 8)
        self.conv8 = conv_conv_pool(96, [32,32], pool=False, use_bn=use_bn)
        
        self.up9 = upconv_2D_MB(16, 9)
        self.conv9 = conv_conv_pool(48, [16,16], pool=False, use_bn=use_bn)
        
        self.conv_last = nn.Conv2d(
            16,
            3, (1,1)
        )



    def forward(self, input):
        net = input
        use_bn=True
        conv1 = self.conv1(input)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        conv5 = torch.cat((conv5, torch.normal(0, 1, conv5.shape).to(conv5.device)), axis=1)
        
        up6 = torch.cat((self.up6(conv5), conv4), axis=1)
        conv6 = self.conv6(up6)
        
        up7 = torch.cat((self.up6(conv6), conv3), axis=1)
        conv7 = self.conv7(up7)
        
        up8 = torch.cat((self.up6(conv7), conv2), axis=1)
        conv8 = self.conv8(up8)
        
        up9 = torch.cat((self.up6(conv8), conv1), axis=1)
        conv9 = self.conv9(up9)
        
        conv_last = self.conv_last(conv9)
        return torch.sigmoid(conv_last)    
   