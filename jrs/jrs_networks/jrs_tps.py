import torch
from torch import nn
from torch.nn import functional as F
from baseclass.basenet import LoadableModel, store_config_args

import itertools
from jrs_networks.jrs_parts import UnetEncoder
import numpy as np
from tps.tps_grid_gen import TPSGridGen
from tps.grid_sample import grid_sample as my_grid_sample

'''
单纯tps的配准网络
'''

class TPSHead(nn.Module):
    def __init__(self, input_channal,grid_height, grid_width, target_control_points,feat_size=4):#grid_height * grid_width * 2
        super().__init__()
        # Spatial transformer localization-network
        self.feat_size=feat_size


        # self.localization = nn.Sequential(
        #     nn.Conv2d(input_channal, 64, kernel_size=3,padding=1),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 32, kernel_size=3,padding=1),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.ReLU(True)
        # )
        self.localization = nn.Sequential(
            nn.Conv2d(input_channal, input_channal//2, kernel_size=7, padding=0),
            nn.BatchNorm2d(input_channal//2),
            nn.ReLU(True),
            nn.Conv2d(input_channal//2, input_channal//4, kernel_size=5, padding=0),
            nn.BatchNorm2d(input_channal//4),
            nn.ReLU(True),
            # nn.Conv2d(64, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(),
            # nn.ReLU(True),
        )

        num_output=grid_height*grid_width*2
        self.num_output=grid_width*grid_height
        # Regressor for the 3 * 2 affine matrix
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(32* self.feat_size//4 * self.feat_size//4, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, num_output)
        # )
        self.fc_loc = nn.Sequential(
            nn.Linear(input_channal//4* 6 * 6, num_output),
        )
        # Initialize the weights/bias with identity transformation
        # bias = target_control_points.view(-1)
        # self.fc_loc[2].weight.data.fill_(0)
        # self.fc_loc[2].bias.data.copy_(bias)
        bias = target_control_points.view(-1)
        self.fc_loc[0].weight.data.fill_(0)
        self.fc_loc[0].bias.data.copy_(bias)

    def forward(self,x):
        xs = self.localization(x)
        # xs = xs.view(-1, 32* self.feat_size//4 * self.feat_size//4)
        xs = xs.view(x.size(0), -1)
        points = self.fc_loc(xs)
        points = points.view(-1,self.num_output ,2)
        return points


class TPSReg(LoadableModel):
    @store_config_args
    def __init__(self,args):
        super(TPSReg, self).__init__()
        self.args=args
        # inshape=(-1,self.args.image_size,self.args.image_size)
        # encoder_feat=[8, 16, 32]
        # self.conv=nn.Sequential(
        #     ConvBlock(1,encoder_feat[0]//4,5,padding=2),
        #     ConvBlock(encoder_feat[0]//4,encoder_feat[0]//2,3,padding=1))
        # self.unetencoder=AffineEncoder(nb_features=encoder_feat)

        filter = [4, 8, 16, 32, 64]
        self.unetencoder1 = UnetEncoder(args, nb_features=filter)
        self.unetencoder2 = UnetEncoder(args, nb_features=filter)

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.target_control_points = torch.cat([X, Y], dim=1)

        self.tpshead= TPSHead(filter[-1]*2, args.grid_height, args.grid_width, self.target_control_points,feat_size=self.args.image_size//2**(len(filter)-1))

        self.tps = TPSGridGen(args.image_height, args.image_width, self.target_control_points)

    def warp(self,img,source_control_points):
        # self.tps = TPSGridGen(img.size()[2], img.size()[3], self.target_control_points)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(img.size()[0], img.size()[2], img.size()[3], 2)
        transformed_x = my_grid_sample(img, grid)
        return transformed_x

    def forward(self, mv_img,fix_img):

        # conv_fix=self.conv(fix_img)
        enc_mv, _ = self.unetencoder1(mv_img)
        # conv_mv=self.conv(mv_img)
        enc_fixed, _ = self.unetencoder2(fix_img)
        x = torch.cat((enc_mv, enc_fixed), dim=1)
        points=self.tpshead(x)
        # warp_img = self.warp(mv_img,points)
        # warp_lab = self.warp(mv_lab,points)
        return points