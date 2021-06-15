import torch.nn as nn
import numpy as np

FAIRFACE_AE_N_UPSAMPLING_EXTRA = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 5, 7: 5}

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
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Autoencoder(nn.Module):
    def __init__(self, input_nc, output_nc, split_layer, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
        super(Autoencoder, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d
        if split_layer > 6:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=1)]
        else:
            model = [#nn.ReflectionPad2d(1),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]

        n_downsampling = 4 if split_layer < 6 else 2
        # Special case for 9th block of resnet
        #n_downsampling, n_blocks = 0, 0
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                     kernel_size=3, stride=2,
                                                     padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        
        n_upsampling_extra = FAIRFACE_AE_N_UPSAMPLING_EXTRA[split_layer] + 1  # +1 added for celeba split4
        for i in range(n_upsampling_extra):  # add upsampling layers
            model += [nn.ConvTranspose2d(ngf, ngf,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
            if i == 1 or i == 2:
                model += [nn.Conv2d(ngf, ngf,
                                             kernel_size=3, stride=1, padding=0),
                                             norm_layer(ngf), nn.ReLU(True)]#"""

        if split_layer < 3:
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, ngf//2, kernel_size=7, padding=0)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=7, padding=0)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf//4, output_nc, kernel_size=7, padding=0)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=7, padding=0)]
        elif split_layer < 5:
            model += [nn.Conv2d(ngf, ngf//2, kernel_size=9, padding=0)]  # orig kernel_size=11, changed to 9 for celeba split4
            model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=7, padding=0)]  # orig kernel_size=9, changed to 7 for celeba split4
            model += [nn.Conv2d(ngf//4, output_nc, kernel_size=7, padding=0)]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=7, padding=0)]
        elif split_layer ==5:
            model += [nn.Conv2d(ngf, ngf//2, kernel_size=9, padding=0)]
            model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=7, padding=0)]
            model += [nn.Conv2d(ngf//4, output_nc, kernel_size=7, padding=0)]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=7, padding=0)]
        else:
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf, ngf//2, kernel_size=9, padding=0)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=7, padding=0)]
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(ngf//4, output_nc, kernel_size=7, padding=0)]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=7, padding=0)]

        #model += [nn.ReflectionPad2d(3)]
        #model += [nn.Conv2d(ngf, ngf//2, kernel_size=9, padding=0)]
        #model += [nn.ReflectionPad2d(3)]
        #model += [nn.Conv2d(ngf//2, ngf//4, kernel_size=7, padding=0)]
        #model += [nn.ReflectionPad2d(3)]
        #model += [nn.Conv2d(ngf//4, output_nc, kernel_size=7, padding=0)]
        #model += [nn.ReflectionPad2d(3)]
        #model += [nn.Conv2d(output_nc, output_nc, kernel_size=7, padding=0)]

        self.m = nn.Sequential(*model)
    
    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x

class MinimalDecoder(nn.Module):
    def __init__(self, input_nc, output_nc=3, input_dim=None, output_dim=None):
        super(MinimalDecoder, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        model = [nn.Conv2d(input_nc, 1 * output_nc, kernel_size=1)]
        for num in range(upsampling_num):
            model += [nn.ConvTranspose2d(64 * output_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            #model += [nn.Linear(1 * output_nc * input_dim ** 2, output_nc * output_dim ** 2)]
        # self.m = nn.Sequential(*model)
        self.m = torch.nn.DataParallel(*model, device_ids=range(torch.cuda.device_count()))

    def forward(self, x):
        b = x.shape[0]
        x = self.m[0](x)
        x = x.view(b, -1)
        x = self.m[1](x)
        x = x.view(b, 3, 224, 224)
        return self.m(x)
