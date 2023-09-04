from baseclass.basenet import LoadableModel
from baseclass.basenet import store_config_args
import  torch
from jrs_networks.jrs_tps import TPSHead
from jrs_networks.jrs_parts import UnetEncoder, MultiModalityTpsDecoder, FeatureL2Norm, FeatureCorrelation,MultiModality2TpsDecoder, \
    UNetSegDecoder,MultiModality3MTpsDecoder
import itertools
import numpy as np
from tps.grid_sample import grid_sample as my_grid_sample
from tps.tps_grid_gen import TPSGridGen


'''
输入3个模态数据 把skip connection层进行融合，并进行配准与分割
利用encoder层的输出作为融合，然后进行分割
'''

class JRS3TpsSegNet(LoadableModel):
    @store_config_args
    def __init__(self,args):
        super(JRS3TpsSegNet, self).__init__()
        self.args=args
        self.cc_feat=args.cc_feat
        self.normalize_matches=True

        #encoder
        filter=[4,8,16,32,64]

        self.unet_encoder_c0=UnetEncoder(args, nb_features=filter)
        self.unet_encoder_lge=UnetEncoder(args, nb_features=filter)
        self.unet_encoder_t2=UnetEncoder(args, nb_features=filter)

        #STN
        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.target_control_points = torch.cat([X, Y], dim=1)
        if self.cc_feat==True:
            self.tpshead_c0=TPSHead((self.args.image_size // (2 ** (len(filter) - 1))) ** 2, args.grid_height, args.grid_width, self.target_control_points, feat_size=self.args.image_size // 2 ** (len(filter) - 1))
            self.tpshead_t2=TPSHead((self.args.image_size // (2 ** (len(filter) - 1))) ** 2, args.grid_height, args.grid_width, self.target_control_points, feat_size=self.args.image_size // 2 ** (len(filter) - 1))
        else:
            self.tpshead_c0=TPSHead(filter[-1] * 2, args.grid_height, args.grid_width, self.target_control_points, feat_size=self.args.image_size // 2 ** (len(filter) - 1))
            self.tpshead_t2=TPSHead(filter[-1] * 2, args.grid_height, args.grid_width, self.target_control_points, feat_size=self.args.image_size // 2 ** (len(filter) - 1))
        self.tps_gen=torch.nn.ModuleDict()
        for i in range(0,len(filter)):
            feat_h=args.image_height//2**i
            feat_w=args.image_width//2**i
            self.tps_gen[f"{feat_h}*{feat_w}"]=TPSGridGen(feat_h, feat_w, self.target_control_points)

        self.featureL2Norm = FeatureL2Norm()
        self.featureCorrelation = FeatureCorrelation()
        self.relu=torch.nn.ReLU(inplace=True)

        #decoder
        filter.reverse()
        self.unet_decoder_c0=UNetSegDecoder(args, 2, nb_features=filter)
        self.unet_decoder_t2=UNetSegDecoder(args, 2, nb_features=filter)

        filter=[i*3 for i in filter]
        self.mm_decoder=MultiModality3MTpsDecoder(args,2,filter)

    def warp(self, feat, source_control_points,mode="bilinear",padd="zeros"):
        feat_h=feat.size()[2]
        feat_w=feat.size()[3]
        source_coordinate = self.tps_gen[f"{feat_h}*{feat_w}"](source_control_points.cuda())
        grid = source_coordinate.view(feat.size()[0], feat_h, feat_w, 2)
        transformed_x = my_grid_sample(feat, grid,mode=mode,padding=padd)
        return transformed_x


    # def forward(self, c0_img,t2_img, lge_img):
    #
    #     enc_lge, skip_lge = self.unet_encoder_lge(lge_img)
    #     enc_c0, skip_c0 = self.unet_encoder_c0(c0_img)
    #     enc_t2, skip_t2 = self.unet_encoder_c0(t2_img)
    #
    #
    #     x_c0_lge = self.concate_feat(enc_c0, enc_lge)
    #     self.theta_c0 = self.tpshead(x_c0_lge)
    #
    #     x_t2_lge = self.concate_feat(enc_t2, enc_lge)
    #     self.theta_t2 = self.tpshead(x_t2_lge)
    #
    #
    #
    #     warp_enc_c0=self.warp(enc_c0, self.theta_c0)
    #     warp_enc_t2=self.warp(enc_t2, self.theta_t2)
    #
    #     previous=torch.cat([enc_lge,warp_enc_c0,warp_enc_t2],dim=1)
    #     # previous=enc_fix
    #
    #     c0_seg=self.unet_decoder_c0(enc_c0, skip_c0)
    #     t2_seg=self.unet_decoder_t2(enc_t2, skip_t2)
    #
    #     lge_seg=self.mm_decoder(previous, skip_c0,skip_t2, skip_lge, self.theta_c0,self.theta_t2)
    #
    #     # out=torch.sigmoid(out)
    #
    #     return c0_seg,t2_seg,lge_seg, self.theta_c0,self.theta_t2


    def forward(self, modality_A_img, modality_B_img, commspace_img,ret_feat=False):

        enc_lge, skip_lge = self.unet_encoder_lge(commspace_img)
        enc_c0, skip_c0 = self.unet_encoder_c0(modality_A_img)
        enc_t2, skip_t2 = self.unet_encoder_t2(modality_B_img)


        x_c0_lge = self.concate_feat(enc_c0, enc_lge)
        self.theta_modality_A = self.tpshead_c0(x_c0_lge)

        x_t2_lge = self.concate_feat(enc_t2, enc_lge)
        self.theta_modality_B = self.tpshead_t2(x_t2_lge)



        warp_enc_c0=self.warp(enc_c0, self.theta_modality_A)
        warp_enc_t2=self.warp(enc_t2, self.theta_modality_B)

        previous=torch.cat([enc_lge,warp_enc_c0,warp_enc_t2],dim=1)
        # previous=enc_fix

        modalit_A_seg=self.unet_decoder_c0(enc_c0, skip_c0)
        modality_B_seg=self.unet_decoder_t2(enc_t2, skip_t2)

        commonspace_seg=self.mm_decoder(previous, skip_c0, skip_t2, skip_lge, self.theta_modality_A, self.theta_modality_B)

        # out=torch.sigmoid(out)
        if ret_feat==False:
            return modalit_A_seg, modality_B_seg, commonspace_seg, self.theta_modality_A, self.theta_modality_B
        else:
            return skip_c0,skip_t2,skip_lge, self.theta_modality_A, self.theta_modality_B

    def concate_feat(self, feat1, feat2):
        if self.cc_feat:
            feature_A = self.featureL2Norm(feat1)
            feature_B = self.featureL2Norm(feat2)
            # do feature correlation
            x_bssfp_t2 = self.featureCorrelation(feature_A, feature_B)
            # normalize
            if self.normalize_matches:
                x_bssfp_t2 = self.featureL2Norm(self.relu(x_bssfp_t2))
        else:
            x_bssfp_t2 = torch.cat([feat1, feat2], dim=1)
        return x_bssfp_t2
