from torch import nn


def closest(lst, K):
    return min(range(len(lst)), key = lambda i: abs(lst[i]-K))


def up_down_to_closest(patch_size, target_patch_size, inp_nc, target_patch_nc):
    diff = patch_size - target_patch_size
    norm_layer=nn.BatchNorm2d
    use_bias = False

    model = [
                nn.Conv2d(inp_nc, target_patch_nc, kernel_size=3, padding=1, bias=use_bias),
                norm_layer(target_patch_nc),
                nn.ReLU(True),
            ]
    if diff == 0:
        return model
    elif diff > 0:
        layer = [nn.Conv2d(target_patch_nc, target_patch_nc, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(target_patch_nc), nn.ReLU(True)]
    else:
        layer = [nn.Conv2d(target_patch_nc, target_patch_nc,
                           kernel_size=2, stride=1, padding=0),
                           norm_layer(target_patch_nc), nn.ReLU(True)]
    model += diff * [layer]
    return model


def up_to_image(model, upsampling_extra, ngf):
    norm_layer=nn.BatchNorm2d
    use_bias = False
    for i in range(upsampling_extra):
        model += [nn.ConvTranspose2d(ngf, ngf//2,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1, bias=use_bias),
                    norm_layer(ngf//2), nn.ReLU(True)]
        ngf = max(3, ngf//2) # 3 because of RGB assumption

    while ngf > 3:
        model += [
                    nn.Conv2d(ngf, ngf//2, kernel_size=3, padding=1, bias=use_bias),
                    norm_layer(ngf//2),
                    nn.ReLU(True),
                ]
        ngf //= 2
    assert ngf == 3

    return model

class Decoder(nn.Module):
    """ This decoder has been designed with the goal to make it
    as generic as possible. It first generates featuremap sizes
    like [128, 64, 32, 16, 8, 4]. Then similarly it generates
    channels list like [3, 6, 12, 24, 48, 96, 192]. Then it takes
    the patch and puts its size to one of the featuremap size and
    number of channels to one in the list. Finally, it performs
    upscaling to obtain a [bs, 3, x, x] sized image.
    """
    def __init__(self, config):
        super(Decoder, self).__init__()
        input_nc = config["channels"]
        img_size = config["img_size"]
        patch_size = config["patch_size"]
        # Because usually images get downsampled by a factor of 2
        feat_map_size = img_size
        feat_map_list = []
        while (feat_map_size > 4):
            feat_map_list.append(feat_map_size)
            feat_map_size //= 2

        patch_channels = [3] # 3 because of RGB
        while (patch_channels[-1] < input_nc):
            patch_channels.append(patch_channels[-1] * 2)
        # remove the last element to ensure the closest match is smallest
        patch_channels.pop()
        # Find closest feature map
        index = closest(feat_map_list, patch_size)
        index_channel = closest(patch_channels, input_nc)

        # upscale/downscale to closest feature map
        model = up_down_to_closest(patch_size, feat_map_list[index],
                                             input_nc, patch_channels[index_channel])

        # construct the remaining layers for the generative model
        self.model = up_to_image(model, len(feat_map_list[:index]),
                                 patch_channels[index_channel])

        self.m = nn.Sequential(*model)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x
