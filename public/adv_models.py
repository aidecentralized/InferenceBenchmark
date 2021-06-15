import torch
import torch.nn as nn
from torchvision import models


def reconstruction_loss(img1, img2):
    return nn.L1Loss()(img1, img2)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class AdversaryModelGen(nn.Module):
    """ 
    """
    def __init__(self, config):
        super(AdversaryModelGen, self).__init__()
        input_nc = config["channels"]
        output_nc = 3
        ngf = 32
        use_bias = False
        n_blocks = 2
        use_dropout = False
        padding_type = 'reflect'
        norm_layer = nn.BatchNorm2d

        model = [
                    nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True),
                ]
        n_downsampling = 2

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        n_upsampling_extra = 3 - n_downsampling 
        for i in range(n_upsampling_extra):
            model += [nn.ConvTranspose2d(ngf, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
            if False and i == 0:
                model += [nn.Conv2d(ngf, ngf,
                                    kernel_size=2, stride=1, padding=0),
                         norm_layer(ngf), nn.ReLU(True)]
        model += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=3, stride=2,
                                     padding=1, output_padding=1, bias=use_bias)]
        #model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, )
        '''model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ngf//2, kernel_size=7, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf//4, output_nc, kernel_size=5, padding=0)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(output_nc, output_nc, kernel_size=5, padding=0)]'''

        self.m = nn.Sequential(*model)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x

class AdversaryModelPred(nn.Module):
    """ Nothing special about the adversary model,
    it is a standard predictive model. Might update it later
    """
    def __init__(self, config):
        super(AdversaryModelPred, self).__init__()
        self.logits = config["logits"]
        self.split_layer = config["split_layer"]

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(num_ftrs, self.logits))

        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i <= self.split_layer:
                continue
            x = l(x)
        return nn.functional.softmax(x, dim=1)


