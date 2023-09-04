# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:46
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : losses.py
"""

"""

import torch
from torch import nn as nn
from torch.autograd import Function
import math
from torch.nn import functional as F


from tools.torch_op_util2 import GaussianSmoothing



class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

import  numpy as np
def dice_coef(y_true, y_pred):
    smooth = 0.0001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection +smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def Cal_Dice(true,pred,tlabels,plabels):
    assert len(tlabels)==len(plabels),print("labels length not equal ")
    dice=[]
    for i in range(len(tlabels)):
        dice.append([])
        img_true=np.zeros_like(true)
        img_pred = np.zeros_like(pred)
        for label in tlabels[i]:
            img_true[true==label]=1
        for label in plabels[i]:
            img_pred[pred == label] = 1
        print(tlabels[i], plabels[i],dice_coef(img_true,img_pred))
        dice[i].append(dice_coef(img_true,img_pred))
    return dice






# class NCC:
#     """
#     Local (over window) normalized cross correlation loss.
#     """
#
#     def __init__(self, win=None):
#         self.win = win
#
#     def loss(self, y_true, y_pred):
#
#         I = y_true
#         J = y_pred
#
#         # get dimension of volume
#         # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#         ndims = len(list(I.size())) - 2
#         assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#
#         # set window size
#         win = [9] * ndims if self.win is None else self.win
#
#         # compute filters
#         sum_filt = torch.ones([1, 1, *win]).to("cuda")
#
#         pad_no = math.floor(win[0]/2)
#
#         if ndims == 1:
#             stride = (1)
#             padding = (pad_no)
#         elif ndims == 2:
#             stride = (1,1)
#             padding = (pad_no, pad_no)
#         else:
#             stride = (1,1,1)
#             padding = (pad_no, pad_no, pad_no)
#
#         # get convolution function
#         conv_fn = getattr(F, 'conv%dd' % ndims)
#
#         # compute CC squares
#         I2 = I * I
#         J2 = J * J
#         IJ = I * J
#
#         I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
#         J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
#         I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
#         J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
#         IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)
#
#         win_size = np.prod(win)
#         u_I = I_sum / win_size
#         u_J = J_sum / win_size
#
#         cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
#         I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
#         J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
#
#         cc = cross * cross / (I_var * J_var + 1e-5)
#
#         return -torch.mean(cc)

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss. Normalized to window [0,1], with 0 being perfect match.
    We follow the NCC definition from the paper "VoxelMorph: A Learning Framework for Deformable Medical Image Registration",
    which implements it via the coefficient of determination (R2 score).
    This is strictly the squared normalized cross-correlation, or squared cosine similarity.
    NCC over two image pacthes I, J of size N is calculated as
    NCC(I, J) = 1/N * [sum_n=1^N (I_n - mean(I)) * (J_n - mean(J))]^2 / [var(I) * var(J)]
    The output is rescaled to the interval [0..1], best match at 0.
    """

    def __init__(self, window=5):
        super().__init__()
        self.win = window

    def forward(self, y_true, y_pred):
        def compute_local_sums(I, J):
            # calculate squared images
            I2 = I * I
            J2 = J * J
            IJ = I * J

            # take sums
            I_sum = conv_fn(I, filt, stride=stride, padding=padding)
            J_sum = conv_fn(J, filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

            # take means
            win_size = np.prod(filt.shape)
            u_I = I_sum / win_size
            u_J = J_sum / win_size

            # calculate cross corr components
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

            return I_var, J_var, cross

        # get dimension of volume
        ndims = 2
        channels = y_true.shape[1]

        # set filter
        filt = torch.ones(channels, channels, *([self.win] * ndims)).type_as(y_true)

        # get convolution function
        conv_fn = getattr(F, "conv%dd" % ndims)
        stride = 1
        padding = self.win // 2

        # calculate cc
        var0, var1, cross = compute_local_sums(y_true, y_pred)
        cc = cross * cross / (var0 * var1 + 1e-5)

        # mean and invert for minimization
        return -torch.mean(cc)

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target,act=None):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        if act:
            predict = act(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class BinaryGaussianDice(nn.Module):
    def __init__(self, sigma=1, reduction='mean'):
        super(BinaryGaussianDice, self).__init__()
        self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self.gau= GaussianSmoothing(1, kernel_size, sigma, 2)


    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict=self.gau(predict)
        target=self.gau(target)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.epsilon
        den = torch.sum(torch.abs(predict)+ torch.abs(target), dim=1) + self.epsilon

        loss = - 2*num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class MvMMGaussianDice_BK(nn.Module):
    def __init__(self, sigma=1, reduction='mean'):
        super(MvMMGaussianDice, self).__init__()
        self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self._epsilon = 1 / 10 ** (6)
        self.gau= GaussianSmoothing(1, kernel_size, sigma, 2)

    def _ngf(self, image):

        dx = (image[..., 1:, 1:] - image[..., :-1, 1:])
        dy = (image[..., 1:, 1:] - image[..., 1:, :-1])

        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return torch.linalg.norm(F.pad(torch.cat((dx, dy), dim=1) , (0, 1, 0, 1)),dim=1,keepdim=True)

    def forward(self, warped_labs_list:list, target_lab_list,prior):
        outs=[]
        for warped_labs,target_lab in zip(warped_labs_list,target_lab_list):
            n_warp=len(warped_labs)
            warped_labs=[self.gau(i) for i in warped_labs]
            target_lab=self.gau(target_lab)
            # warped_labs = [i.contiguous().view(i.shape[0], -1) for i in warped_labs]
            # target_lab = target_lab.contiguous().view(target_lab.shape[0], -1)

            joint_warp=torch.ones_like(warped_labs[0])
            for i in warped_labs:
                joint_warp = joint_warp * i
            joint_pro=joint_warp*target_lab

            outs.append(joint_pro)


            # mask_target=self._ngf(target_lab)
            # mask_warped_lab=[self._ngf(i) for i in warped_labs]
            # joint_mask = torch.ones_like(warped_labs[0])
            # for i in mask_warped_lab:
            #     joint_mask=joint_mask*i
            # joint_mask=joint_mask*mask_target
            #
            # out=joint_mask*joint_pro
            # outs.append(out)

        final=torch.zeros_like(outs[0])
        for p,i in zip(prior,outs):
            final=final+i*p
        # final=-torch.log(final)

        return -final.mean()

class MvMMGaussianDice_from_TF(nn.Module):
    '''
    完全按照MvMM的tensorflow代码实现
    '''
    def __init__(self, sigma=1, reduction='mean'):
        super(MvMMGaussianDice_from_TF, self).__init__()
        # self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self._epsilon = 1e-5
        self.gau= GaussianSmoothing(4, kernel_size, sigma, 2)

    def _ngf(self, image):

        dx = (image[..., 1:, 1:] - image[..., :-1, 1:])
        dy = (image[..., 1:, 1:] - image[..., 1:, :-1])
        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)
        norm =F.pad(norm, (0, 1, 0, 1))
        mask=norm>1e-3
        mask=mask.float()

        return mask

    def forward(self, warped_labs:list, target_lab,prior=[0,1,0,0.1]):
        outs=[]
        prior=torch.tensor(prior)
        prior=prior.repeat(4, 256, 256, 1)
        prior=torch.transpose(prior,1,3)

        warped_labs=[self.gau(i) for i in warped_labs]
        target_lab=self.gau(target_lab)
        # warped_labs = [i.contiguous().view(i.shape[0], -1) for i in warped_labs]
        # target_lab = target_lab.contiguous().view(target_lab.shape[0], -1)

        joint_warp=torch.ones_like(warped_labs[0])
        for i in warped_labs:
            joint_warp = joint_warp * i
        prob_product=joint_warp*target_lab*prior.cuda()

        mask_target=self._ngf(target_lab)
        mask_warped_lab=[self._ngf(i) for i in warped_labs]


        mask_product = torch.ones_like(warped_labs[0])
        for i in mask_warped_lab:
            mask_product=mask_product*i
        mask_product=mask_product*mask_target


        all_product=mask_product*prob_product
        sum_mask=(all_product>self._epsilon)
        sum_mask=sum_mask.any(1)

        ll=torch.log(torch.clip(all_product.sum(dim=[1]),self._epsilon,1))

        all_product=((ll*sum_mask).sum(dim=[1,2])+self._epsilon)/(sum_mask.float().sum(dim=[1,2])+self._epsilon)

        return -all_product.mean()



    # final=torch.zeros_like(outs[0])
    # for p,i in zip(prior,outs):
    #     final=final+i*p
    # final=torch.clip(final,1e-5,1)
    # final=torch.log(final+self._epsilon)
    #
    #
    # return -final.mean()

class MvMMGaussianDice(nn.Module):
    '''
    这个可以训练出T2的结果，但是C0不行，
    lr=1e-5
    '''
    def __init__(self, sigma=1,ch=4, reduction='mean'):
        super(MvMMGaussianDice, self).__init__()
        self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self._epsilon = 1e-5
        self.gau= GaussianSmoothing(ch, kernel_size, sigma, 2)

    def _ngf(self, image):

        dx = (image[..., 1:, 1:] - image[..., :-1, 1:])
        dy = (image[..., 1:, 1:] - image[..., 1:, :-1])
        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)
        norm =F.pad(norm, (0, 1, 0, 1))
        mask=norm>1e-3
        mask=mask.float()

        return mask

    def forward(self, warped_labs:list, target_lab,prior=[0,1,0,0.1]):
        outs=[]
        prior=torch.tensor(prior).cuda()


        warped_labs=[self.gau(i) for i in warped_labs]
        target_lab=self.gau(target_lab)
        # warped_labs = [i.contiguous().view(i.shape[0], -1) for i in warped_labs]
        # target_lab = target_lab.contiguous().view(target_lab.shape[0], -1)

        joint_warp=torch.ones_like(warped_labs[0])

        for i,p in zip(warped_labs,prior):
            joint_warp = joint_warp * i
        joint_pro=joint_warp*target_lab

        total=torch.zeros_like(target_lab)
        for i in warped_labs:
            total=torch.abs(i)
        total=total+target_lab

        out=(joint_pro.sum(dim=[2,3])+self._epsilon)/(total.sum(dim=[2,3])+self._epsilon)
        out=out.mean(dim=0)

        return -(out*prior.cuda()).sum()

class MvMMGaussianDice_V2(nn.Module):
    def __init__(self, sigma=1, reduction='mean'):
        super(MvMMGaussianDice, self).__init__()
        self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self._epsilon = 1e-5
        self.gau= GaussianSmoothing(4, kernel_size, sigma, 2)

    def _ngf(self, image):

        dx = (image[..., 1:, 1:] - image[..., :-1, 1:])
        dy = (image[..., 1:, 1:] - image[..., 1:, :-1])
        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)
        norm =F.pad(norm, (0, 1, 0, 1))
        mask=norm>1e-3
        mask=mask.float()

        return mask

    def forward(self, warped_labs:list, target_lab,prior=[0,1,0,0.1]):
        outs=[]
        prior=torch.tensor(prior).cuda()


        warped_labs=[self.gau(i) for i in warped_labs]
        target_lab=self.gau(target_lab)
        # warped_labs = [i.contiguous().view(i.shape[0], -1) for i in warped_labs]
        # target_lab = target_lab.contiguous().view(target_lab.shape[0], -1)

        joint_warp=torch.ones_like(warped_labs[0])

        for i,p in zip(warped_labs,prior):
            joint_warp = joint_warp * i
        joint_pro=joint_warp*target_lab

        total=torch.zeros_like(target_lab)
        for i in warped_labs:
            total=torch.abs(i)
        total=total+target_lab

        out=(joint_pro.sum(dim=[2,3])+self._epsilon)/(total.sum(dim=[2,3])+self._epsilon)
        out=out.mean(dim=0)

        return -(out*prior.cuda()).sum()


class GaussianNGF(nn.Module):
    def __init__(self, sigma=1, reduction='mean'):
        super(GaussianNGF, self).__init__()
        self.epsilon = 1
        self.sigma=sigma
        self.reduction = reduction
        kernel_size=int(3*sigma)*2+1
        self._dim=2
        self._epsilon= 1/10**(6)
        self.gau= GaussianSmoothing(2, kernel_size, sigma, 2)

    def _ngf(self, image):

        dx = (image[..., 1:, 1:] - image[..., :-1, 1:])
        dy = (image[..., 1:, 1:] - image[..., 1:, :-1])

        norm = torch.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(torch.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict=self._ngf(predict)
        target=self._ngf(target)
        predict=self.gau(predict)
        target=self.gau(target)
        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)
        loss = 0
        for dim in range(self._dim):
            loss = loss + predict[:, dim, ...] * target[:, dim, ...]

        loss = -torch.abs(loss)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


# 这个高斯模糊只针对在 单通道的二值图中才可以使用
class BinaryMultiScaleGaussianDice(nn.Module):
    def __init__(self, sigmas=[1,3,5,7]):
        super(BinaryMultiScaleGaussianDice, self).__init__()
        self.gau_dices=[]
        self.sigmas=sigmas
        for sig in sigmas:
            self.gau_dices.append(BinaryGaussianDice(sig))


    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        loss = 0
        for gau_dice in self.gau_dices:
            c_loss=gau_dice(predict,target)
            # print(c_loss)
            loss=loss+c_loss

        return loss/len(self.sigmas)

class BinaryDiceLoss(nn.Module):
    # """
    # N-D dice for segmentation
    # """
    # def __init__(self):
    #     super().__init__()
    #
    # def forward(self, y_true, y_pred):
    #     ndims = len(list(y_pred.size())) - 2
    #     vol_axes = list(range(2, ndims+2))
    #     top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    #     bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    #     dice = torch.mean(top / bottom)
    #     return -dice
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        # den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        den = torch.sum(torch.abs(predict)+ torch.abs(target), dim=1) + self.smooth

        loss = 1 - 2*num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class MultiScaleGaussianDiceLoss(nn.Module):
    """MultiScaleGaussianDiceLoss loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as MultiScaleGaussianDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(MultiScaleGaussianDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryMultiScaleGaussianDice(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i].unsqueeze(1), target[:, i].unsqueeze(1))
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]



# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

# class SoftDiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#     def forward(self, logits, targets):
#         num = targets.size(0)
#         smooth = 1
#
#         probs = F.sigmoid(logits)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :])
            dice = 2. * (inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :])
            dice = 2. * (inter + 1) / (union + 1)
            dice_sum += dice
            # print(dice)
        return 1 - dice_sum / class_num


class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):

        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)

        return 1 - 2. * inter / union


def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
    union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
    dice = (2. * inter + 0.00001) / (union + 1)
    return dice


def T(logits, targets):
    return torch.sum(targets[:, 2, :, :])


def P(logits, targets):
    return torch.sum(logits[:, 2, :, :])


def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :] * logits[:, 2, :, :])

#
# class JacobianDeterminant(nn.Module):
#     def __init__(self, reduction="mean", preserve_size=False):
#         super(JacobianDeterminant, self).__init__()
#         self.idty = torchreg.nn.Identity()
#         self.reduction = reduction
#         self.ndims = torchreg.settings.get_ndims()
#         self.preserve_size = preserve_size
#
#     def forward(self, flow):
#         """
#         calculates the area of each pixel after the flow is applied
#         """
#
#         def determinant_2d(x, y):
#             return x[:, [0]] * y[:, [1]] - x[:, [1]] * y[:, [0]]
#
#         def determinant_3d(x, y, z):
#             return (x[:, [0]] * y[:, [1]] * z[:, [2]] +
#                     x[:, [2]] * y[:, [0]] * z[:, [1]] +
#                     x[:, [1]] * y[:, [2]] * z[:, [0]] -
#                     x[:, [2]] * y[:, [1]] * z[:, [0]] -
#                     x[:, [1]] * y[:, [0]] * z[:, [2]] -
#                     x[:, [0]] * y[:, [2]] * z[:, [1]])
#
#         if self.preserve_size:
#             flow = F.pad(flow, [0, 1] * self.ndims, mode="replicate")
#
#         # map to target domain
#         transform = flow + self.idty(flow)
#
#         # get finite differences
#         if self.ndims == 2:
#             dx = torch.abs(transform[:, :, 1:, :-1] - transform[:, :, :-1, :-1])
#             dy = torch.abs(transform[:, :, :-1, 1:] - transform[:, :, :-1, :-1])
#             jacdet = determinant_2d(dx, dy)
#         elif self.ndims == 3:
#             dx = torch.abs(transform[:, :, 1:, :-1, :-1] - transform[:, :, :-1, :-1, :-1])
#             dy = torch.abs(transform[:, :, :-1, 1:, :-1] - transform[:, :, :-1, :-1, :-1])
#             dz = torch.abs(transform[:, :, :-1, :-1, 1:] - transform[:, :, :-1, :-1, :-1])
#             jacdet = determinant_3d(dx, dy, dz)
#
#         if self.reduction == "none":
#             return jacdet
#         elif self.reduction == "mean":
#             return torch.mean(jacdet)
#         elif self.reduction == "sum":
#             return torch.sum(jacdet)
#
