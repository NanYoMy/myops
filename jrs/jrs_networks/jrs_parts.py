import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvBlock(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvBlockV2(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(in_channels//2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class AffineDown(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, in_channels, out_channels, stride=1,kernel_size=3):
        super().__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            ConvBlock(in_channels, out_channels, kernel_size, padding=1),
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out

class UnetDown(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, in_channels, out_channels, stride=2,kernel_size=3):
        super().__init__()
        self.maxpool_conv=nn.Sequential(
            nn.MaxPool2d(2, stride=stride),
            DoubleConvBlock(in_channels, out_channels, kernel_size, 1),
        )

    def forward(self, x):
        out = self.maxpool_conv(x)
        return out

class UNetUp(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, in_channels, out_channels, stride=1, bilinear=True):
        super().__init__()

        # if bilinear==True:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConvBlock(in_channels*2,out_channels)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        #     self.conv = DoubleConvBlock(in_channels,out_channels)

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DeepSupeeUNetUp(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self, in_channels, up_channels,out_channels=None, stride=1, bilinear=True):
        super().__init__()

        # if bilinear==True:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConvBlock(in_channels*2,out_channels)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        #     self.conv = DoubleConvBlock(in_channels,out_channels)

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, up_channels)
        self.deepsuper = torch.nn.Conv2d(up_channels, out_channels,kernel_size=1)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        main=self.conv(x)
        ds=self.deepsuper(main)
        return main, ds

class ChannalFusion(nn.Module):
    def __init__(self, in_channels, out_channels,reduction=4,bilinear=True):
        super(ChannalFusion, self).__init__()
        # if bilinear==True:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        # self.conv = DoubleConvBlock(in_channels * 2, out_channels)
        self.conv = DoubleConvBlock(in_channels , out_channels)

    def forward(self, prev,i_0,i_1,i_2):
        prev=self.up(prev)
        #**** channal attention start *****
        main=torch.cat([prev,i_0,i_1,i_2],dim=1)
        b, c, _, _ = main.size()
        y = self.avg_pool(main).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        main=main * y.expand_as(main)
        #**** channal attention end *****
        return self.conv(main)

class AffineEncoder(nn.Module):

    def __init__(self, nb_features=[8, 16, 32]):
        super().__init__()
        self.nb_feature=nb_features
        prev_nf=nb_features[0]//2
        self.encoders=nn.ModuleList()
        ks=[3]*len(nb_features)
        for k,nf in zip(ks,nb_features):
            self.encoders.append(AffineDown(prev_nf, nf, stride=1, kernel_size=k))
            prev_nf=nf
    def forward(self,input):
        out_feat=[]
        out_feat.append(input)
        for enc in self.encoders:
            x=enc(out_feat[-1])
            out_feat.append(x)
        return out_feat[-1],out_feat[1:]

class UnetEncoder(nn.Module):

    def __init__(self,args, nb_features=[4,8,16,32,64],input=1):
        super().__init__()
        self.args=args
        self.nb_feature=nb_features
        prev_nf=input
        #初始的卷积层
        self.conv = nn.Sequential(ConvBlock(prev_nf, nb_features[0]//2 , 7,padding=3),
                                  ConvBlock(nb_features[0] // 2, nb_features[0], 5,padding=2))
        self.downs=nn.ModuleList()
        prev_nf=nb_features[0]
        #
        ks=[3]*len(nb_features[1:])
        for k,nf in zip(ks,nb_features[1:]):
            self.downs.append(UnetDown(prev_nf, nf, kernel_size=k))
            prev_nf=nf
    def forward(self,input):
        out_feat=[]
        x=self.conv(input)
        out_feat.append(x)
        for down in self.downs:
            x=down(out_feat[-1])
            out_feat.append(x)
        return out_feat[-1],out_feat[:-1]

class UNetSegDecoder(nn.Module):
    def __init__(self,args,n_classes, nb_features=[64,32,16,8,4]):
        super().__init__()
        self.ups=nn.ModuleList()
        self.args=args
        for nf in nb_features[:-1]:
            self.ups.append(UNetUp(nf, nf//2))

        self.outc=nn.Sequential(ConvBlock(nb_features[-1], nb_features[-1]//2),
                                nn.Conv2d(nb_features[-1]//2,n_classes,kernel_size=1))


    def forward(self, previous, skip_connection):
        x=previous
        for i,up in enumerate(self.ups):
            x=up(x, skip_connection[-(i+1)])
        x=self.outc(x)
        return x

class UNetDeepSuperSegDecoder(nn.Module):
    def __init__(self, args, n_classes, nb_features=[64, 32, 16, 8, 4]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.args = args
        for nf in nb_features[:-1]:
            self.ups.append(DeepSupeeUNetUp(nf, nf //2, nf //2))

        self.outc = nn.Sequential(ConvBlock(nb_features[-1], nb_features[-1] // 2),
                                  nn.Conv2d(nb_features[-1] // 2, n_classes, kernel_size=1))

    def forward(self, previous, skip_connection):
        x = previous
        deepout=[]
        for i, up in enumerate(self.ups):
            x, out= up(x, skip_connection[-(i + 1)])
            deepout.append(out)
        x = self.outc(x)
        return x,deepout

class UNetRegDecoder(nn.Module):
    def __init__(self,args, nb_features=[64,32,16,8,4]):
        super().__init__()
        self.ups=nn.ModuleList()

        for nf in nb_features[:-1]:
            self.ups.append(UNetUp(nf, nf//2))

        self.outc=nn.Sequential(ConvBlock(nb_features[-1]//2,
                                          nb_features[-1]//2),
                                nn.Conv2d(nb_features[-1]//2,2,kernel_size=3))


    def forward(self, previous, skip_connection):
        x=previous
        for i,up in enumerate(self.ups):
            x=up(x, skip_connection[-(i+1)])
        x=self.outc(x)
        return x

class MultiModalityAffDecoder(nn.Module):
    def __init__(self, args, filter=[64, 32, 16, 8, 4]):
        super().__init__()
        self.args=args
        # self.ups=nn.ModuleList()
        self.fusions=nn.ModuleList()
        for nf in filter[:-1]:
            self.fusions.append(ChannalFusion(nf,nf//2))
            # self.ups.append(UNetUp(nf, nf//2))

        self.outc=nn.Sequential(ConvBlock(filter[-1] // 2,
                                          filter[-1] // 2),
                                nn.Conv2d(filter[-1] // 2, args.nb_class, kernel_size=1))

    def forward(self, previous, skips_de_, skips_t2, skips_c0_, theta_t2, theta_c0):

        # for i,(up, fus) in enumerate(zip(self.ups,self.fusions)):
        for i,fus in enumerate(self.fusions):
            skip_de=skips_de_[-i - 1]
            skip_t2=skips_t2[-i-1]
            skip_c0=skips_c0_[-i - 1]

            grid_t2 = F.affine_grid(theta_t2, skip_t2.size())
            skip_t2 = F.grid_sample(skip_t2, grid_t2)

            grid_c0 = F.affine_grid(theta_c0, skip_c0.size())
            skip_c0 = F.grid_sample(skip_c0, grid_c0)

            previous=fus(previous,skip_de,skip_t2,skip_c0)
            # previous=up(previous)


        return self.outc(previous)


from tps.tps_grid_gen import TPSGridGen
import itertools
import numpy as np
from tps.grid_sample import grid_sample as my_grid_sample

class CrossModality2Fusion(nn.Module):

    def __init__(self, in_channels, out_channels,reduction=4,bilinear=True):
        super().__init__()
        # if bilinear==True:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(in_channels), #先不进行norm
            nn.Sigmoid()
        )
        self.conv = DoubleConvBlock(in_channels , out_channels)

    def forward(self, prev, skip, cross_modality_skip):

        prev=self.up(prev)
        cross_modality_skip=self.att(cross_modality_skip)
        # prev=prev*cross_modality_skip
        skip=skip*cross_modality_skip
        main=torch.cat([prev, skip,cross_modality_skip], dim=1)
        return self.conv(main)


class MultiModality2TpsDecoder(nn.Module):
    def __init__(self, args,out_c, filter=[128, 64, 32, 16, 8]):
        super().__init__()
        self.args=args
        # self.ups=nn.ModuleList()
        self.fusions=nn.ModuleList()

        for nf in filter[:-1]:
            self.fusions.append(CrossModality2Fusion(nf,nf//2))
            # self.ups.append(UNetUp(nf, nf//2))

        self.outc = nn.Sequential(DoubleConvBlock(filter[-1],filter[-1] // 2),
                                  nn.Conv2d(filter[-1] // 2, out_c, kernel_size=1))

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.tps_gen=nn.ModuleDict()
        self.target_control_points = torch.cat([X, Y], dim=1)
        for i in range(0,len(filter)):
            feat_h=args.image_height//2**i
            feat_w=args.image_width//2**i
            self.tps_gen[f"{feat_h}*{feat_w}"]=TPSGridGen(feat_h, feat_w, self.target_control_points)

    def warp(self, feat, source_control_points):
        feat_h = feat.size()[2]
        feat_w = feat.size()[3]
        source_coordinate = self.tps_gen[f"{feat_h}*{feat_w}"](source_control_points)
        grid = source_coordinate.view(feat.size()[0], feat_h, feat_w, 2)
        transformed_x = my_grid_sample(feat, grid)
        return transformed_x

    def forward(self, previous, skips_mv, skips_fix, theta):

        # for i,(up, fus) in enumerate(zip(self.ups,self.fusions)):
        for i,fus in enumerate(self.fusions):
            skip_mv=skips_mv[-i - 1]
            skip_fix=skips_fix[-i - 1]
            skip_mv=self.warp(skip_mv, theta)
            previous=fus(previous,skip_mv,skip_fix)

        return self.outc(previous)


class CrossModality3MFusion(nn.Module):

    def __init__(self, in_channels, out_channels,reduction=4,bilinear=True):
        super().__init__()
        # if bilinear==True:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels , in_channels//2, kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.att = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(in_channels), #先不进行norm
            nn.Sigmoid()
        )
        self.conv = DoubleConvBlock(in_channels , out_channels)


    # def forward(self, prev, c0_skip,t2_skip,de_skip):
    #
    #     prev=self.up(prev)
    #
    #     att_c0_skip=self.att(c0_skip)
    #     att_t2_skip=self.att(t2_skip)
    #     att=torch.mean(torch.stack([att_c0_skip,att_t2_skip]), 0)
    #     # prev=prev*cross_modality_skip
    #     de_skip= de_skip * att
    #     # de_skip= de_skip * c0_skip * t2_skip
    #     main=torch.cat([prev,c0_skip,t2_skip,de_skip], dim=1)
    #     return self.conv(main)

    def forward(self, prev, c0_skip,t2_skip,de_skip):

        prev=self.up(prev)

        att_c0_skip=self.att(c0_skip)
        att_t2_skip=self.att(t2_skip)
        att=torch.mean(torch.stack([att_c0_skip,att_t2_skip]), 0)
        # prev=prev*cross_modality_skip
        de_skip= de_skip * att
        # de_skip= de_skip * c0_skip * t2_skip
        main=torch.cat([prev,de_skip,att_c0_skip,att_t2_skip], dim=1)
        return self.conv(main)

class MultiModality3MTpsDecoder(nn.Module):
    def __init__(self, args,out_c, filter=[64*3, 32*3, 16*3, 8*3, 4*3]):
        super().__init__()
        self.args=args
        # self.ups=nn.ModuleList()
        self.fusions=nn.ModuleList()

        for nf in filter[:-1]:
            self.fusions.append(CrossModality3MFusion(nf, nf // 2))
            # self.ups.append(UNetUp(nf, nf//2))

        self.outc = nn.Sequential(DoubleConvBlock(filter[-1],filter[-1] // 2),
                                  nn.Conv2d(filter[-1] // 2, out_c, kernel_size=1))

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.tps_gen=nn.ModuleDict()
        self.target_control_points = torch.cat([X, Y], dim=1)
        for i in range(0,len(filter)):
            feat_h=args.image_height//2**i
            feat_w=args.image_width//2**i
            self.tps_gen[f"{feat_h}*{feat_w}"]=TPSGridGen(feat_h, feat_w, self.target_control_points)

    def warp(self, feat, source_control_points):
        feat_h = feat.size()[2]
        feat_w = feat.size()[3]
        source_coordinate = self.tps_gen[f"{feat_h}*{feat_w}"](source_control_points)
        grid = source_coordinate.view(feat.size()[0], feat_h, feat_w, 2)
        transformed_x = my_grid_sample(feat, grid)
        return transformed_x

    def forward(self, previous, skips_c0, skips_t2, skips_de, theta_c0,theta_t2):

        for i,fus in enumerate(self.fusions):
            skip_c0 = skips_c0[-i - 1]
            skip_t2 = skips_t2[-i - 1]
            skip_de = skips_de[-i - 1]
            skip_c0=self.warp(skip_c0, theta_c0)
            skip_t2=self.warp(skip_t2, theta_t2)
            previous=fus(previous,skip_c0,skip_t2,skip_de)

        return self.outc(previous)


class MultiModalityTpsDecoder(MultiModality2TpsDecoder):
    def __init__(self, args, filter=[64, 32, 16, 8, 4]):
        super().__init__()
        self.args=args
        # self.ups=nn.ModuleList()
        self.fusions=nn.ModuleList()
        for nf in filter[:-1]:
            self.fusions.append(ChannalFusion(nf,nf//2))
            # self.ups.append(UNetUp(nf, nf//2))

        self.outc=nn.Sequential(DoubleConvBlock(filter[-1],
                                          filter[-1] // 2),
                                nn.Conv2d(filter[-1] // 2, args.nb_class, kernel_size=1))

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.tps_gen={}
        self.target_control_points = torch.cat([X, Y], dim=1)
        for i in range(0,len(filter)):
            feat_h=args.image_height//2**i
            feat_w=args.image_width//2**i
            self.tps_gen[f"{feat_h}*{feat_w}"]=TPSGridGen(feat_h, feat_w, self.target_control_points)


    def forward(self, previous, skips_de, skips_t2, skips_c0, theta_t2, theta_c0):

        # for i,(up, fus) in enumerate(zip(self.ups,self.fusions)):
        for i,fus in enumerate(self.fusions):
            skip_de=skips_de[-i - 1]
            skip_t2=skips_t2[-i-1]
            skip_c0=skips_c0[-i - 1]
            skip_t2=self.warp(skip_t2,theta_t2)
            # grid_t2 = F.affine_grid(theta_t2, skip_t2.size())
            # skip_t2 = F.grid_sample(skip_t2, grid_t2)
            # grid_lge = F.affine_grid(theta_lge, skip_lge.size())
            # skip_lge = F.grid_sample(skip_lge, grid_lge)
            skip_c0 = self.warp(skip_c0, theta_c0)

            previous=fus(previous,skip_de,skip_t2,skip_c0)
            # previous=up(previous)


        return self.outc(previous)


class MultiModalityBinaryDecoder(MultiModalityAffDecoder):
    def __init__(self, args, filter=[64, 32, 16, 8, 4]):
        super().__init__(args,filter)
        self.outc=nn.Sequential(ConvBlock(filter[-1] // 2,
                                          filter[-1] // 2),
                                nn.Conv2d(filter[-1] // 2, 1, kernel_size=1))


class MultiModalityFusionSegDecoder(MultiModalityAffDecoder):
    def __init__(self, args, filter=[64, 32, 16, 8, 4]):
        super().__init__(args, filter)


    def forward(self,previous,skips_bssfp,skips_t2,skips_lge):

        # for i,(up, fus) in enumerate(zip(self.ups,self.fusions)):
        for i,fus in enumerate(self.fusions):
            skip_t2=skips_t2[-i-1]
            skip_lge=skips_lge[-i-1]
            skip_bssfp=skips_bssfp[-i-1]

            previous=fus(previous,skip_bssfp,skip_t2,skip_lge)
            # previous=up(previous)

        return self.outc(previous)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class RAGate(nn.Module):
    def __init__(self, args,in_channels, out_channels,reduction=4,bilinear=False):
        super(RAGate, self).__init__()
        if bilinear==True:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down={}
        for i in range(5):
            w=args.image_size//(2**i)
            h=args.image_size//(2**i)
            self.down[f'{w}x{h}']=nn.AdaptiveAvgPool2d((w, h))

        self.conv = DoubleConvBlock(in_channels * 2, out_channels)
        self.sig=nn.Sigmoid()

    def forward(self, prev,skip,constraint):
        prev=self.up(prev)
        #cross modality contraint
        main=torch.cat([prev,skip],dim=1)
        if constraint is not None:
            # constraint=torchvision.transforms.Resize(constraint,(w, h))
            b, c, w, h = main.size()
            att = self.sig(constraint)
            # att=constraint
            att=self.down[f'{w}x{h}'](att)

            main=main * att.expand_as(main)
        return self.conv(main)

class RADecoder(nn.Module):
    def __init__(self, args, filter=[64, 32, 16, 8, 4]):
        super().__init__()
        self.args=args
        # self.ups=nn.ModuleList()
        self.fusions=nn.ModuleList()
        for nf in filter[:-1]:
            self.fusions.append(RAGate(args,nf, nf // 2))
            # self.ups.append(UNetUp(nf, nf//2))

        self.outc=nn.Sequential(ConvBlock(filter[-1] // 2, filter[-1] // 2),
                                nn.Conv2d(filter[-1] // 2, args.nb_class, kernel_size=1),
                                )

    def forward(self,previous,skips,constraint):

        # for i,(up, fus) in enumerate(zip(self.ups,self.fusions)):
        for i,fus in enumerate(self.fusions):
            skip=skips[-i-1]

            previous=fus(previous,skip,constraint)
            # previous=up(previous)


        return self.outc(previous)