import math
import numbers

import torch
from torch import nn as nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        self.kernal_size=kernel_size
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # print(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        # spatial_pad = [self.kernel.size(2) // 2,
        #                self.kernel.size(2) // 2,
        #                self.kernel.size(3) // 2,
        #                self.kernel.size(3) // 2,
        #                self.kernel.size(4) // 2,
        #                self.kernel.size(4) // 2]
        # out_ch: int = 6 if self.order == 2 else 3
        # # return F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c).view(b, c, out_ch, d, h, w)
        # res=F.conv3d(F.pad(input, spatial_pad, 'replicate'), kernel, padding=0, groups=c)
        self.spatial_pad= [self.kernal_size[0] // 2, self.kernal_size[0] // 2,self.kernal_size[1]//2,self.kernal_size[1]//2] if dim==2 else [self.kernal_size[0] // 2, self.kernal_size[0] // 2, self.kernal_size[1] // 2,self.kernal_size[1] // 2,   self.kernal_size[2] // 2,  self.kernal_size[2] // 2]
        if dim == 1:
            self.conv = F.conv1d

        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
            # self.spatial_pad = [self.kernal_size[0] // 2,
            #            self.kernal_size[0] // 2,
            #            self.kernal_size[1] // 2,
            #            self.kernal_size[1] // 2,
            #            self.kernal_size[2] // 2,
            #            self.kernal_size[2] // 2]
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # print(self.kernal_size)
        input=F.pad(input, self.spatial_pad, 'replicate')
        if torch.cuda.is_available():
            return self.conv(input, weight=self.weight.to('cuda'), groups=self.groups)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups)