import numpy as np
import torch
import torch.nn as nn
from skimage import transform as transform
import SimpleITK as sitk
import cv2
from skimage.filters import gaussian
from skimage import transform,exposure

def make_torch_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target

def make_numpy_one_hot(y, num_classes=2):
    y=np.transpose(y,[1,2,0])
    y=to_catagory(y,num_classes)
    y=np.transpose(y,[2,0,1])
    return y

def to_catagory(y,num_classes=2):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def un_torch_one_hot(data):
    return torch.argmax(data,dim=1,keepdim=True)

def convert_gray_2_rgb(tensor):
    c1=torch.where(tensor==1,torch.full_like(tensor,255),torch.full_like(tensor,0))
    c2=torch.where(tensor==2,torch.full_like(tensor,255),torch.full_like(tensor,0))
    c3=torch.where(tensor==3,torch.full_like(tensor,255),torch.full_like(tensor,0))
    rgb=torch.cat([c1,c2,c3],dim=1).float()/255
    return rgb

def convert_binary_2_rgb(tensor):
    if tensor.shape[1]==1:
        rgb=make_torch_one_hot(tensor,3)
    elif tensor.shape[1]==3:
        rgb=tensor
    else:
        rgb=make_torch_one_hot(torch.argmax(tensor,dim=1,keepdim=True),3)
    return rgb

from batchgenerators.augmentations.utils import resize_segmentation
def downsample_seg_for_ds_transform2(seg, ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)), order=0, cval=0, axes=None):
    if axes is None:
        axes = list(range(2, len(seg.shape)))
    output = []
    for s in ds_scales:
        if all([i == 1 for i in s]):
            output.append(seg)
        else:
            new_shape = np.array(seg.shape).astype(float)
            for i, a in enumerate(axes):
                new_shape[a] *= s[i]
            new_shape = np.round(new_shape).astype(int)
            out_seg = np.zeros(new_shape, dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(seg.shape[1]):
                    out_seg[b, c] = resize_segmentation(seg[b, c], new_shape[2:], order, cval)
            output.append(out_seg)
    return output


class SkimageOP_Base():
    def __init__(self):
        pass

    # def resize_image(self,img,mask=None,size=(128, 128)):
    #     img = transform.resize(img, size)
    #     if mask!=None:
    #         mask = transform.resize(mask, size, order=0)
    #         return img,mask
    #     else:
    #         return img

    def equhisto(self,img):
        return exposure.equalize_hist(img)

    def usm(self,img):

        img = img * 1.0
        gauss_out = gaussian(img, sigma=5, multichannel=True)

        # alpha 0 - 5
        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / np.max(img)

        # 饱和处理
        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        return img_out

    def gamma_correction(self,img,rang=(7,15),prob=0.3):
        rand1 = np.random.rand()
        if rand1<prob:
            rand2=np.random.randint(7,15)/10.0
            img = exposure.adjust_gamma(img,rand2)
        return img


    # def resize(self,img,size,order=None):
    #     img = np.squeeze(img).astype(np.float)
    #     img =transform.resize(img,size,order=order)
    #     return img

    def resize(self,image,size,order=1):
        image = np.squeeze(image)
        if order==0:
            img=cv2.resize(image,(size[1],size[0]),interpolation=cv2.INTER_NEAREST)
        else:
            img=cv2.resize(image,(size[1],size[0]))
        return img

    def normalize_image_label(self,img,mask,size=(256,256),clip=False):
        img = img.astype(np.float)
        mask = mask.astype(np.float)#一定要转化成float
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        if clip==True:
            img =self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img, mask

    def normalize_intensity_image(self,img,size=(256,256),clip=False):
        img = img.astype(np.float)

        img = transform.resize(img, size)
        if clip==True:
            img =self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img

    def normalize_multiseq(self, img1, img2, img3, mask, size=(256, 256)):

        img1=img1.astype(np.float)
        img2=img2.astype(np.float)
        img3=img3.astype(np.float)
        mask=mask.astype(np.float)

        img1 = transform.resize(img1, size)
        img2 = transform.resize(img2, size)
        img3 = transform.resize(img3, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.

        img1 = (self.intensity_rescale(img1))  ###########
        img2 = (self.intensity_rescale(img2))
        img3 = (self.intensity_rescale(img3))
        return img1, img2, img3, mask

    def normalize_image(self,img,mask,size=(256,256),clip=False):
        img = img.astype(np.float)
        mask = mask.astype(np.float)
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        if clip==True:
            img =self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img, mask

    def normalize_non_zero_region(self,img):

        print("region normal")
        mask=np.where(img!=0,1,0)
        cnt=np.sum(mask)
        mean=np.sum(img*mask)/cnt
        sig2=np.sum((img-mean)*(img-mean))/cnt
        new_img=(img-mean)/np.sqrt(sig2)
        return new_img*mask

    def resize_images(self,img,mask,size):
        img = img.astype(np.float)
        mask = mask.astype(np.float)
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        return img, mask


    def aug_multiseq(self, c0, de, t2, mask):
        c0, de, t2, mask = self.random_flip_multiseq(c0, de, t2, mask)
        c0, c0_mask = self.random_affine(c0, mask)
        de, de_mask = self.random_affine(de, mask)
        t2, t2_mask = self.random_affine(t2, mask)
        return c0, c0_mask, de, de_mask, t2, t2_mask

    def aug_multiseq(self, c0, t2, de, c0_lab,t2_lab,de_lab,aug_ratio=0.3):
        c0, t2, de, c0_lab,t2_lab,de_lab = self.random_flip_multiseq(c0, t2, de, c0_lab,t2_lab,de_lab)
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            c0, c0_lab = self.random_affine(c0, c0_lab)
            t2, t2_lab = self.random_affine(t2, t2_lab)
            de, de_lab = self.random_affine(de, de_lab)
        return c0.copy(), t2.copy(), de.copy(), c0_lab.copy(), t2_lab.copy(), de_lab.copy()

    def rand_small_affine_multiseq(self,c0, c0_lab, de, de_lab, t2, t2_lab,aug_ratio=0.3):
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            c0, c0_lab = self.random_samll_affine(c0, c0_lab)
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            t2, t2_lab = self.random_samll_affine(t2, t2_lab)
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            de, de_lab = self.random_samll_affine(de, de_lab)
        return c0,c0_lab,  de, de_lab, t2, t2_lab

    def aug_multiseqV2(self, c0, t2, de, c0_lab, t2_lab, de_lab, aug_ratio=0.3):

        # 统一进行flip
        c0, t2, de, c0_lab, t2_lab, de_lab = self.random_flip_multiseqV2(c0, t2, de, c0_lab, t2_lab, de_lab)
        # 统一进行大范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_large_affine_multiseq( c0, c0_lab, de, de_lab, t2, t2_lab,aug_ratio)
        # 分别进行小范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_small_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab, 0.1)

        return c0.copy(), t2.copy(), de.copy(), c0_lab.copy(), t2_lab.copy(), de_lab.copy()

    def random_mid_affine_with_lab(self, img, lab):
        rd_scale = np.random.uniform(0.9, 1.1)

        rd_translate_x = np.random.uniform(-0.1, 0.1) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.1,0.1) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 36, np.pi / 36)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.affine_trans(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)


    def rand_mid_affine_multiseq(self,c0, lab_c0, de, lab_de, t2, lab_t2,aug_ratio=0.3):
        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            c0, lab_c0 = self.random_mid_affine_with_lab(c0, lab_c0)

        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            t2, lab_t2 = self.random_mid_affine_with_lab(t2, lab_t2)

        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            de, lab_de = self.random_mid_affine_with_lab(de, lab_de)
        return c0, lab_c0, de, lab_de, t2, lab_t2
    def aug_multiseqV3(self, c0, t2, de, c0_lab, t2_lab, de_lab, aug_ratio=0.3):

        # 统一进行flip
        c0, t2, de, c0_lab, t2_lab, de_lab = self.random_flip_multiseqV2(c0, t2, de, c0_lab, t2_lab, de_lab)
        # 统一进行大范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_large_affine_multiseq( c0, c0_lab, de, de_lab, t2, t2_lab,0.1)
        # 分别进行中范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_mid_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab, 0.3)

        return c0.copy(), t2.copy(), de.copy(), c0_lab.copy(), t2_lab.copy(), de_lab.copy()

    def rand_large_affine_multiseq(self,  c0, c0_lab, de, de_lab, t2, t2_lab,aug_ratio):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo

            rd_scale = np.random.uniform(0.9, 1.2)
            # print(f'scale:{rd_scale}')
            rd_translate_x = np.random.uniform(-0.1, 0.1) * c0.shape[0]
            rd_translate_y = np.random.uniform(-0.1, 0.1) * c0.shape[1]
            # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
            rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)

            c0, c0_lab = self.affine_trans(c0, c0_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            t2, t2_lab = self.affine_trans(t2, t2_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            de, de_lab = self.affine_trans(de, de_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
        return c0, c0_lab, de, de_lab, t2, t2_lab

    def aug_img(self, img, mask):
        img, mask = self.random_flip(img, mask)
        img, mask = self.random_affine(img, mask)
        return img, mask

    def convert_array_2_torch(self, img):
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img).float()
        return img

    def random_affine_multiseq(self, img1, img2, img3, mask):
        rot = 0  # np.random.randint(0, 360)
        tra = np.random.randint(-5, 5, size=2)
        she = np.random.uniform(-0.1, 0.1)
        img1 = transform.warp(img1, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img2 = transform.warp(img2, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img3 = transform.warp(img3, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        mask = transform.warp(mask, transform.AffineTransform(translation=tra, rotation=rot, shear=she), order=0)
        return img1, img2, img3, mask

    def random_affine(self, img, mask):
        rd_scale = np.random.uniform(0.9, 1.2)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.2, 0.2) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.2, 0.2) * img.shape[1]
        rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        # rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
        return self.affine_trans(img, mask, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    # def random_samll_affine(self, img, mask):
    #     rd_scale = np.random.uniform(0.95, 1.05)
    #     # print(f'scale:{rd_scale}')
    #     rd_translate_x = np.random.uniform(-0.03, 0.03) * img.shape[0]
    #     rd_translate_y = np.random.uniform(-0.03,0.03) * img.shape[1]
    #     rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
    #     return self.affine_trans(img, mask, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def random_samll_affine(self, img, lab):
        rd_scale = np.random.uniform(0.95, 1.05)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.03, 0.03) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.03,0.03) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
        return self.affine_trans(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)


    def affine_trans(self, img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y):
        shift_y, shift_x = (np.array(img.shape) - 1) / 2.
        tf_center_fw = transform.AffineTransform(scale=1, translation=[-shift_x, -shift_y], rotation=0)
        tf_affine = transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y),
                                              rotation=rd_rotate)
        tf_center_bk = transform.AffineTransform(scale=1, translation=[shift_x, shift_y], rotation=0)
        trans_img = transform.warp(img, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=3)
        trans_mask = transform.warp(lab, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        return trans_img, trans_mask

    def random_deform(self,img,mask):
        pass

    def zscore_mask(self, data, mask):
        indices = np.where(mask > 0)
        mean = data[indices].mean()
        std = data[indices].std()
        data[indices] = (data[indices] - mean) / std
        # 其他的值保持为0
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def intensity_rescale_mask(self, data, mask):
        indices = np.where(mask > 0)
        # mean = data[indices].mean()
        max = data[indices].max()
        min = data[indices].min()
        range = max - min
        data[indices] = (data[indices] - min) / range
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def zscore(self, data):
        mu = np.mean(data)
        sigma = np.std(data)
        data = (data - mu) / sigma
        # print(np.max(data))
        return data

    def intensity_rescale(self, data):
        data=data.astype("float")
        range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / range

        return data

    # 当需要分割的是pathology的时候，不要使用clip_intensity_rescale,因为一般pathology也是高亮的灰度值
    def clip_intensity_rescale(self, data, low_percent=5, high_percent=95):
        # p0 = data.min().astype('float')
        low_bound = np.percentile(data, low_percent)
        high_bound = np.percentile(data, high_percent)
        # p100 = data.max().astype('float')
        data=np.where(data>high_bound,high_bound,data)
        data=np.where(data < low_bound, low_bound, data)
        range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / range
        if range==0:
            print("error")
        return data

    def random_rotate(self, img1, img2, img3, mask):
        randa = np.random.randint(1, 360)
        img1 = transform.rotate(img1, randa)
        img2 = transform.rotate(img2, randa)
        img3 = transform.rotate(img3, randa)
        mask = transform.rotate(mask, randa)
        return img1, img2, img3, mask

    def random_flip_multiseq(self, img1, img2, img3, mask):
        rand1 = np.random.rand()
        # rand1=torch.rand().item()
        rand2 = np.random.rand()
        # rand2=torch.rand().item()
        if rand1 > 0.5:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            img3 = np.flipud(img3)
            mask = np.flipud(mask)
        if rand2 > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            img3 = np.fliplr(img3)
            mask = np.fliplr(mask)
        return img1, img2, img3, mask

    def random_flip_multiseqV2(self, img1, img2, img3, mask1,mask2,mask3):
        rand1 = np.random.rand()
        # rand1=torch.rand().item()
        rand2 = np.random.rand()
        # rand2=torch.rand().item()
        if rand1 > 0.5:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            img3 = np.flipud(img3)
            mask1 = np.flipud(mask1)
            mask2 = np.flipud(mask2)
            mask3 = np.flipud(mask3)
        if rand2 > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            img3 = np.fliplr(img3)
            mask1 = np.fliplr(mask1)
            mask2 = np.fliplr(mask2)
            mask3 = np.fliplr(mask3)
        return img1, img2, img3, mask1,mask2,mask3

    def random_flip(self, img, mask):
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        if rand1 > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if rand2 > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask

    # def random_crop(self, img1, img2, img3, mask, size=(128, 128)):
    #     shape = img1.shape
    #     s1 = np.random.randint(0, shape[0] - size[0])
    #     s2 = np.random.randint(0, shape[1] - size[1])
    #     img1 = img1[s1:s1 + size[0], s2:s2 + size[1]]
    #     img2 = img2[s1:s1 + size[0], s2:s2 + size[1]]
    #     img3 = img3[s1:s1 + size[0], s2:s2 + size[1]]
    #     mask = mask[s1:s1 + size[0], s2:s2 + size[1]]
    #     return img1, img2, img3, mask
    #
    # def random_step(self, img1, img2, img3, mask, size=(123, 123)):
    #     shape = img1.shape
    #     s1 = np.random.randint(0, shape[0] - size[0])
    #     s2 = np.random.randint(0, shape[1] - size[1])
    #     nimg1 = np.zeros_like(img1)
    #     nimg2 = np.zeros_like(img2)
    #     nimg3 = np.zeros_like(img2)
    #     nmask = np.zeros_like(mask)
    #     img1, img2, img3, mask = self.random_crop(img1, img2, img3, mask, size)
    #     nimg1[s1: s1 + size[0], s2: s2 + size[1]] = img1
    #     nimg2[s1: s1 + size[0], s2: s2 + size[1]] = img2
    #     nimg3[s1: s1 + size[0], s2: s2 + size[1]] = img3
    #     nmask[s1: s1 + size[0], s2: s2 + size[1]] = mask
    #
    #     return nimg1, nimg2, nimg3, nmask
class SkimageOP_jrs_Pathology(SkimageOP_Base):
    def __init__(self):
        super().__init__()
    def random_flip_multiseqV3(self, img1, img2, ms1, ms2, lab1, lab2):
        rand1 = np.random.rand()
        # rand1=torch.rand().item()
        rand2 = np.random.rand()
        # rand2=torch.rand().item()
        if rand1 > 0.5:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            ms1 = np.flipud(ms1)
            ms2 = np.flipud(ms2)
            lab1 = np.flipud(lab1)
            lab2 = np.flipud(lab2)
        if rand2 > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            ms1 = np.fliplr(ms1)
            ms2 = np.fliplr(ms2)
            lab1 = np.fliplr(lab1)
            lab2 = np.fliplr(lab2)
        return img1, img2, ms1, ms2, lab1, lab2
    def affine_trans_with_mask(self, img, mask,lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y):
        shift_y, shift_x = (np.array(img.shape) - 1) / 2.
        tf_center_fw = transform.AffineTransform(scale=1, translation=[-shift_x, -shift_y], rotation=0)
        tf_affine = transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y),
                                              rotation=rd_rotate)
        tf_center_bk = transform.AffineTransform(scale=1, translation=[shift_x, shift_y], rotation=0)
        trans_img = transform.warp(img, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=3)
        trans_mask = transform.warp(mask, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        trans_lab = transform.warp(lab, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        return trans_img, trans_mask,trans_lab
    def random_samll_affine_with_mask(self, img, mask,lab):
        rd_scale = np.random.uniform(0.999, 1.001)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.01, 0.01) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.01,0.01) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 360, np.pi / 360)
        return self.affine_trans_with_mask(img, mask,lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def rand_small_affine_multiseqV3(self,c0,de,mask_c0,mask_de,lab_c0,lab_de,aug_ratio=0.3):
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            c0, mask_c0,lab_c0 = self.random_samll_affine_with_mask(c0, mask_c0,lab_c0)
        rand1 = np.random.rand()
        if rand1>aug_ratio: # 按照一定比例进行data augo
            de,mask_de, lab_de = self.random_samll_affine_with_mask(de,mask_de, lab_de)
        return c0,de,mask_c0,mask_de,lab_c0,lab_de,
    def rand_large_affine_multiseqV3(self,  c0,de,mask_c0,mask_de,lab_c0,lab_de,aug_ratio):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo

            rd_scale = np.random.uniform(0.9, 1.2)
            # print(f'scale:{rd_scale}')
            rd_translate_x = np.random.uniform(-0.1, 0.1) * c0.shape[0]
            rd_translate_y = np.random.uniform(-0.1, 0.1) * c0.shape[1]
            # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
            rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)

            c0, mask_c0, lab_c0= self.affine_trans_with_mask(c0, mask_c0, lab_c0,rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            de, mask_de, lab_de, = self.affine_trans_with_mask(de, mask_de, lab_de,rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
        return  c0,de,mask_c0,mask_de,lab_c0,lab_de,
    def aug_multiseqV3(self,c0,de,mask_c0,mask_de,lab_c0,lab_de,aug_ratio=0.3):
        c0,de,mask_c0,mask_de,lab_c0,lab_de = self.random_flip_multiseqV3(c0,de,mask_c0,mask_de,lab_c0,lab_de)

        c0,de,mask_c0,mask_de,lab_c0,lab_de = self.rand_large_affine_multiseqV3(c0,de,mask_c0,mask_de,lab_c0,lab_de, aug_ratio)
        c0,de,mask_c0,mask_de,lab_c0,lab_de = self.rand_small_affine_multiseqV3(c0,de,mask_c0,mask_de,lab_c0,lab_de, 0.1)

        return c0.copy(),  de.copy(), mask_c0.copy(), mask_de.copy(), lab_c0.copy(), lab_de.copy()


class SkimageOP_MyoPS20(SkimageOP_Base):
    def __init__(self):
        super().__init__()
    def random_flip_multiseq(self, img1, img2,img3, lab1, lab2,lab3):
        rand1 = np.random.rand()
        # rand1=torch.rand().item()
        rand2 = np.random.rand()
        # rand2=torch.rand().item()
        if rand1 > 0.5:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            img3 = np.flipud(img3)

            lab1 = np.flipud(lab1)
            lab2 = np.flipud(lab2)
            lab3 = np.flipud(lab3)
        if rand2 > 0.5:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            img3 = np.fliplr(img3)

            lab1 = np.fliplr(lab1)
            lab2 = np.fliplr(lab2)
            lab3 = np.fliplr(lab3)
        return img1, img2, img3, lab1, lab2,lab3
    def affine_trans_with_lab(self, img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y):
        shift_y, shift_x = (np.array(img.shape) - 1) / 2.
        tf_center_fw = transform.AffineTransform(scale=1, translation=[-shift_x, -shift_y], rotation=0)
        tf_affine = transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y),
                                              rotation=rd_rotate)
        tf_center_bk = transform.AffineTransform(scale=1, translation=[shift_x, shift_y], rotation=0)
        trans_img = transform.warp(img, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=3)

        trans_lab = transform.warp(lab, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        return trans_img, trans_lab
    def random_samll_affine_with_lab(self, img, lab):
        rd_scale = np.random.uniform(0.999, 1.001)

        rd_translate_x = np.random.uniform(-0.01, 0.01) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.01,0.01) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 360, np.pi / 360)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.affine_trans_with_lab(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
    def random_mid_affine_with_lab(self, img, lab):
        rd_scale = np.random.uniform(0.999, 1.111)

        rd_translate_x = np.random.uniform(-0.1, 0.1) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.1,0.1) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 36, np.pi / 36)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.affine_trans_with_lab(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    # def random_samll_affine_with_lab(self, img, lab):
    #     rd_scale = np.random.uniform(0.95, 1.05)
    #
    #     rd_translate_x = np.random.uniform(-0.05, 0.05) * img.shape[0]
    #     rd_translate_y = np.random.uniform(-0.05,0.05) * img.shape[1]
    #     # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
    #     rd_rotate = np.random.uniform(-np.pi / 360, np.pi / 360)
    #     # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
    #     return self.affine_trans_with_lab(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def rand_mid_affine_multiseq_myops(self,c0,t2,de,lab_c0,lab_t2,lab_de,aug_ratio=0.3):
        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            c0, lab_c0 = self.random_mid_affine_with_lab(c0, lab_c0)

        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            t2, lab_t2 = self.random_mid_affine_with_lab(t2, lab_t2)

        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            de, lab_de = self.random_mid_affine_with_lab(de, lab_de)
        return c0, t2, de, lab_c0, lab_t2, lab_de,
    def rand_small_affine_multiseq(self,c0,t2,de,lab_c0,lab_t2,lab_de,aug_ratio=0.3):
        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            c0, lab_c0 = self.random_samll_affine_with_lab(c0, lab_c0)

        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            t2, lab_t2 = self.random_samll_affine_with_lab(t2, lab_t2)

        rand = np.random.rand()
        if rand>aug_ratio: # 按照一定比例进行data augo
            de, lab_de = self.random_samll_affine_with_lab(de, lab_de)
        return c0,t2,de,lab_c0,lab_t2,lab_de,
    def rand_large_affine_multiseq(self,  c0,t2,de,lab_c0,lab_t2,lab_de,aug_ratio):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo

            rd_scale = np.random.uniform(0.9, 1.2)
            # print(f'scale:{rd_scale}')
            rd_translate_x = np.random.uniform(-0.1, 0.1) * c0.shape[0]
            rd_translate_y = np.random.uniform(-0.1, 0.1) * c0.shape[1]
            # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
            rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)

            c0,  lab_c0= self.affine_trans_with_lab(c0, lab_c0, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            t2,  lab_t2= self.affine_trans_with_lab(t2, lab_t2, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            de,  lab_de, = self.affine_trans_with_lab(de, lab_de, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
        return  c0,t2,de,lab_c0,lab_t2,lab_de,
    def aug_multiseq(self,c0,t2,de,lab_c0,lab_t2,lab_de,aug_ratio=0.3):
        c0,t2,de,lab_c0,lab_t2,lab_de = self.random_flip_multiseq(c0,t2,de,lab_c0,lab_t2,lab_de)

        c0,t2,de,lab_c0,lab_t2,lab_de = self.rand_large_affine_multiseq(c0,t2,de,lab_c0,lab_t2,lab_de, aug_ratio)
        c0,t2,de,lab_c0,lab_t2,lab_de= self.rand_small_affine_multiseq(c0,t2,de,lab_c0,lab_t2,lab_de, 0.1)

        return c0.copy(),t2.copy(), de.copy(),  lab_c0.copy(),lab_t2.copy(), lab_de.copy()

    def aug_multiseq_myops(self,c0,t2,de,lab_c0,lab_t2,lab_de,aug_ratio=0.3):
        c0,t2,de,lab_c0,lab_t2,lab_de = self.random_flip_multiseq(c0,t2,de,lab_c0,lab_t2,lab_de)

        c0,t2,de,lab_c0,lab_t2,lab_de = self.rand_large_affine_multiseq(c0,t2,de,lab_c0,lab_t2,lab_de, aug_ratio)
        c0,t2,de,lab_c0,lab_t2,lab_de= self.rand_mid_affine_multiseq_myops(c0,t2,de,lab_c0,lab_t2,lab_de, 0.1)

        return c0.copy(),t2.copy(), de.copy(),  lab_c0.copy(),lab_t2.copy(), lab_de.copy()


from PIL import Image
class SkimageOP_MSCMR(SkimageOP_MyoPS20):
    def __init__(self):

        super().__init__()


    def resize(self,image,size,order=1):
        image = np.squeeze(image)
        if order==0:
            img=cv2.resize(image,(size[1],size[0]),interpolation=cv2.INTER_NEAREST)
        else:
            img=cv2.resize(image,(size[1],size[0]))
        return img


from baseclass.medicalimage import ImgType,Modality,MedicalImage



class SkimageOP_RJ_PSN(SkimageOP_Base):
    def __init__(self):
        super().__init__()



    def random_flipud(self,img_data:MedicalImage):
        img_data.img=np.flipud(img_data.img).copy()
        img_data.gt_lab=np.flipud(img_data.gt_lab).copy()
        img_data.prior=np.flipud(img_data.prior).copy()

        return img_data

    def random_fliplr(self,img_data:MedicalImage):
        img_data.img=np.fliplr(img_data.img).copy()
        img_data.gt_lab=np.fliplr(img_data.gt_lab).copy()
        img_data.prior=np.fliplr(img_data.prior).copy()

        return img_data

    def random_affine_trans(self, img_data:MedicalImage, rd_rotate, rd_scale, rd_translate_x, rd_translate_y):
        shift_y, shift_x = (np.array(img_data.img.shape) - 1) / 2.
        tf_center_fw = transform.AffineTransform(scale=1, translation=[-shift_x, -shift_y], rotation=0)
        tf_affine = transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y),
                                              rotation=rd_rotate)
        tf_center_bk = transform.AffineTransform(scale=1, translation=[shift_x, shift_y], rotation=0)
        img_data.img = transform.warp(img_data.img, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=3)
        img_data.gt_lab = transform.warp(img_data.gt_lab, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        img_data.prior = transform.warp(img_data.prior, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        return img_data

    def random_samll_affine(self, imgdata):
        rd_scale = np.random.uniform(0.999, 1.001)

        rd_translate_x = np.random.uniform(-0.01, 0.01) * imgdata.img.shape[0]
        rd_translate_y = np.random.uniform(-0.01,0.01) *  imgdata.img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 360, np.pi / 360)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.random_affine_trans(imgdata, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def random_mid_affine(self, imgdata):
        rd_scale = np.random.uniform(0.999, 1.111)

        rd_translate_x = np.random.uniform(-0.1, 0.1) * imgdata.img.shape[0]
        rd_translate_y = np.random.uniform(-0.1,0.1) * imgdata.img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 36, np.pi / 36)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.random_affine_trans(imgdata, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def random_large_affine(self, imgdata):
        rd_scale = np.random.uniform(0.9, 1.2)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.1, 0.1) * imgdata.img.shape[0]
        rd_translate_y = np.random.uniform(-0.1, 0.1) * imgdata.img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.random_affine_trans(imgdata, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)


    def random_flip_multiseq(self, img_datas):

        rand1 = np.random.rand()
        rand2 = np.random.rand()
        if rand1 > 0.5:
            for k in img_datas.keys():
                img_datas[k]=self.random_flipud(img_datas[k])
        if rand2 > 0.5:
            for k in img_datas.keys():
                img_datas[k]=self.random_fliplr(img_datas[k])
        return img_datas

    def prep_normalize(self,datas,size=[128,128]):

        for k in datas.keys():
            data=datas[k]

            data.img=self.usm(data.img)

            data.img=self.intensity_rescale(data.img)
            # data.img=self.zscore(data.img)


            data.img=self.resize(data.img,size)
            data.prior=self.resize(data.prior,size)
            data.gt_lab=self.resize(data.gt_lab,size,order=0)
        return datas




    def rand_large_affine_multiseq(self, datas,aug_ratio):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo

            rd_scale = np.random.uniform(0.9, 1.2)
            # print(f'scale:{rd_scale}')
            rd_translate_x = np.random.uniform(-0.1, 0.1) * datas["c0"].img.shape[0]
            rd_translate_y = np.random.uniform(-0.1, 0.1) * datas["c0"].img.shape[1]
            # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
            rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)

            for k in datas.keys():

                datas[k] = self.random_affine_trans(datas[k], rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

        return datas



    def augment(self, datas,aug_ratio=0.3):
        datas = self.random_flip_multiseq(datas)
        datas = self.rand_large_affine_multiseq(datas, aug_ratio)
        return datas



class SKimage_Single_modality():
    def __init__(self):
        pass

    # def resize_image(self,img,mask=None,size=(128, 128)):
    #     img = transform.resize(img, size)
    #     if mask!=None:
    #         mask = transform.resize(mask, size, order=0)
    #         return img,mask
    #     else:
    #         return img

    def equhisto(self, img):
        return exposure.equalize_hist(img)

    def usm(self, img):

        img = img * 1.0
        gauss_out = gaussian(img, sigma=5, multichannel=True)

        # alpha 0 - 5
        alpha = 1.5
        img_out = (img - gauss_out) * alpha + img

        img_out = img_out / np.max(img)

        # 饱和处理
        mask_1 = img_out < 0
        mask_2 = img_out > 1

        img_out = img_out * (1 - mask_1)
        img_out = img_out * (1 - mask_2) + mask_2
        return img_out

    def gamma_correction(self, img, rang=(7, 15), prob=0.3):
        rand1 = np.random.rand()
        if rand1 < prob:
            rand2 = np.random.randint(7, 15) / 10.0
            img = exposure.adjust_gamma(img, rand2)
        return img

    # def resize(self,img,size,order=None):
    #     img = np.squeeze(img).astype(np.float)
    #     img =transform.resize(img,size,order=order)
    #     return img

    def resize(self, image, size, order=1):
        image = np.squeeze(image)
        if order == 0:
            img = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            img = cv2.resize(image, (size[1], size[0]))
        return img

    def normalize_image_label(self, img, mask, size=(256, 256), clip=False):
        img = img.astype(np.float)
        mask = mask.astype(np.float)  # 一定要转化成float
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        if clip == True:
            img = self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img, mask

    def normalize_intensity_image(self, img, size=(256, 256), clip=False):
        img = img.astype(np.float)

        img = transform.resize(img, size)
        if clip == True:
            img = self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img

    def normalize_multiseq(self, img1, img2, img3, mask, size=(256, 256)):

        img1 = img1.astype(np.float)
        img2 = img2.astype(np.float)
        img3 = img3.astype(np.float)
        mask = mask.astype(np.float)

        img1 = transform.resize(img1, size)
        img2 = transform.resize(img2, size)
        img3 = transform.resize(img3, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.

        img1 = (self.intensity_rescale(img1))  ###########
        img2 = (self.intensity_rescale(img2))
        img3 = (self.intensity_rescale(img3))
        return img1, img2, img3, mask

    def normalize_image(self, img, mask, size=(256, 256), clip=False):
        img = img.astype(np.float)
        mask = mask.astype(np.float)
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        if clip == True:
            img = self.clip_intensity_rescale(img)
        else:
            img = self.intensity_rescale(img)  ###########
        return img, mask

    def normalize_non_zero_region(self, img):

        print("region normal")
        mask = np.where(img != 0, 1, 0)
        cnt = np.sum(mask)
        mean = np.sum(img * mask) / cnt
        sig2 = np.sum((img - mean) * (img - mean)) / cnt
        new_img = (img - mean) / np.sqrt(sig2)
        return new_img * mask

    def resize_images(self, img, mask, size):
        img = img.astype(np.float)
        mask = mask.astype(np.float)
        img = transform.resize(img, size)
        mask = transform.resize(mask, size, order=0)  # 这里有bug.
        return img, mask



    def aug_multiseq(self, c0,  c0_lab, aug_ratio=0.3):
        # c0, c0_lab = self.random_flip_multiseq(c0,  c0_lab)
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo
            c0, c0_lab = self.random_affine(c0, c0_lab)

        return c0.copy(),  c0_lab.copy()

    def rand_small_affine_multiseq(self, c0, c0_lab, de, de_lab, t2, t2_lab, aug_ratio=0.3):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo
            c0, c0_lab = self.random_samll_affine(c0, c0_lab)
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo
            t2, t2_lab = self.random_samll_affine(t2, t2_lab)
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo
            de, de_lab = self.random_samll_affine(de, de_lab)
        return c0, c0_lab, de, de_lab, t2, t2_lab

    def aug_multiseqV2(self, c0, t2, de, c0_lab, t2_lab, de_lab, aug_ratio=0.3):

        # 统一进行flip
        c0, t2, de, c0_lab, t2_lab, de_lab = self.random_flip_multiseqV2(c0, t2, de, c0_lab, t2_lab, de_lab)
        # 统一进行大范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_large_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab,
                                                                             aug_ratio)
        # 分别进行小范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_small_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab,
                                                                             0.1)

        return c0.copy(), t2.copy(), de.copy(), c0_lab.copy(), t2_lab.copy(), de_lab.copy()

    def random_mid_affine_with_lab(self, img, lab):
        rd_scale = np.random.uniform(0.9, 1.1)

        rd_translate_x = np.random.uniform(-0.1, 0.1) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.1, 0.1) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 36, np.pi / 36)
        # print(f"small {rd_scale},{rd_rotate},{rd_translate_x},{rd_translate_y}")
        return self.affine_trans(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def rand_mid_affine_multiseq(self, c0, lab_c0, de, lab_de, t2, lab_t2, aug_ratio=0.3):
        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            c0, lab_c0 = self.random_mid_affine_with_lab(c0, lab_c0)

        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            t2, lab_t2 = self.random_mid_affine_with_lab(t2, lab_t2)

        rand = np.random.rand()
        if rand > aug_ratio:  # 按照一定比例进行data augo
            de, lab_de = self.random_mid_affine_with_lab(de, lab_de)
        return c0, lab_c0, de, lab_de, t2, lab_t2

    def aug_multiseqV3(self, c0, t2, de, c0_lab, t2_lab, de_lab, aug_ratio=0.3):

        # 统一进行flip
        c0, t2, de, c0_lab, t2_lab, de_lab = self.random_flip_multiseqV2(c0, t2, de, c0_lab, t2_lab, de_lab)
        # 统一进行大范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_large_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab,
                                                                             0.1)
        # 分别进行中范围的affine
        c0, c0_lab, de, de_lab, t2, t2_lab = self.rand_mid_affine_multiseq(c0, c0_lab, de, de_lab, t2, t2_lab, 0.3)

        return c0.copy(), t2.copy(), de.copy(), c0_lab.copy(), t2_lab.copy(), de_lab.copy()

    def rand_large_affine_multiseq(self, c0, c0_lab, de, de_lab, t2, t2_lab, aug_ratio):
        rand1 = np.random.rand()
        if rand1 > aug_ratio:  # 按照一定比例进行data augo

            rd_scale = np.random.uniform(0.9, 1.2)
            # print(f'scale:{rd_scale}')
            rd_translate_x = np.random.uniform(-0.1, 0.1) * c0.shape[0]
            rd_translate_y = np.random.uniform(-0.1, 0.1) * c0.shape[1]
            # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
            rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)

            c0, c0_lab = self.affine_trans(c0, c0_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            t2, t2_lab = self.affine_trans(t2, t2_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
            de, de_lab = self.affine_trans(de, de_lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)
        return c0, c0_lab, de, de_lab, t2, t2_lab

    def aug_img(self, img, mask):
        img, mask = self.random_flip(img, mask)
        img, mask = self.random_affine(img, mask)
        return img, mask

    def convert_array_2_torch(self, img):
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img).float()
        return img

    def random_affine_multiseq(self, img1, img2, img3, mask):
        rot = 0  # np.random.randint(0, 360)
        tra = np.random.randint(-5, 5, size=2)
        she = np.random.uniform(-0.1, 0.1)
        img1 = transform.warp(img1, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img2 = transform.warp(img2, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        img3 = transform.warp(img3, transform.AffineTransform(translation=tra, rotation=rot, shear=she))
        mask = transform.warp(mask, transform.AffineTransform(translation=tra, rotation=rot, shear=she), order=0)
        return img1, img2, img3, mask

    def random_affine(self, img, mask):
        rd_scale = np.random.uniform(0.9, 1.2)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.2, 0.2) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.2, 0.2) * img.shape[1]
        rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        # rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
        return self.affine_trans(img, mask, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    # def random_samll_affine(self, img, mask):
    #     rd_scale = np.random.uniform(0.95, 1.05)
    #     # print(f'scale:{rd_scale}')
    #     rd_translate_x = np.random.uniform(-0.03, 0.03) * img.shape[0]
    #     rd_translate_y = np.random.uniform(-0.03,0.03) * img.shape[1]
    #     rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
    #     return self.affine_trans(img, mask, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def random_samll_affine(self, img, lab):
        rd_scale = np.random.uniform(0.95, 1.05)
        # print(f'scale:{rd_scale}')
        rd_translate_x = np.random.uniform(-0.03, 0.03) * img.shape[0]
        rd_translate_y = np.random.uniform(-0.03, 0.03) * img.shape[1]
        # rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
        rd_rotate = np.random.uniform(-np.pi / 50, np.pi / 50)
        return self.affine_trans(img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y)

    def affine_trans(self, img, lab, rd_rotate, rd_scale, rd_translate_x, rd_translate_y):
        shift_y, shift_x = (np.array(img.shape) - 1) / 2.
        tf_center_fw = transform.AffineTransform(scale=1, translation=[-shift_x, -shift_y], rotation=0)
        tf_affine = transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y),
                                              rotation=rd_rotate)
        tf_center_bk = transform.AffineTransform(scale=1, translation=[shift_x, shift_y], rotation=0)
        trans_img = transform.warp(img, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=3)
        trans_mask = transform.warp(lab, (tf_center_fw + (tf_affine + tf_center_bk)).inverse, order=0)
        return trans_img, trans_mask

    def random_deform(self, img, mask):
        pass

    def zscore_mask(self, data, mask):
        indices = np.where(mask > 0)
        mean = data[indices].mean()
        std = data[indices].std()
        data[indices] = (data[indices] - mean) / std
        # 其他的值保持为0
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def intensity_rescale_mask(self, data, mask):
        indices = np.where(mask > 0)
        # mean = data[indices].mean()
        max = data[indices].max()
        min = data[indices].min()
        range = max - min
        data[indices] = (data[indices] - min) / range
        indices = np.where(mask <= 0)
        data[indices] = 0
        return data

    def zscore(self, data):
        mu = np.mean(data)
        sigma = np.std(data)
        data = (data - mu) / sigma
        # print(np.max(data))
        return data

    def intensity_rescale(self, data):
        data = data.astype("float")
        range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / range

        return data

    # 当需要分割的是pathology的时候，不要使用clip_intensity_rescale,因为一般pathology也是高亮的灰度值
    def clip_intensity_rescale(self, data, low_percent=5, high_percent=95):
        # p0 = data.min().astype('float')
        low_bound = np.percentile(data, low_percent)
        high_bound = np.percentile(data, high_percent)
        # p100 = data.max().astype('float')
        data = np.where(data > high_bound, high_bound, data)
        data = np.where(data < low_bound, low_bound, data)
        range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / range
        if range == 0:
            print("error")
        return data

    def random_rotate(self, img1, img2, img3, mask):
        randa = np.random.randint(1, 360)
        img1 = transform.rotate(img1, randa)
        img2 = transform.rotate(img2, randa)
        img3 = transform.rotate(img3, randa)
        mask = transform.rotate(mask, randa)
        return img1, img2, img3, mask



    def random_flip_multiseq(self, img1,  mask1):
        rand1 = np.random.rand()
        # rand1=torch.rand().item()
        rand2 = np.random.rand()
        # rand2=torch.rand().item()
        if rand1 > 0.5:
            img1 = np.flipud(img1)

            mask1 = np.flipud(mask1)

        if rand2 > 0.5:
            img1 = np.fliplr(img1)

            mask1 = np.fliplr(mask1)

        return img1,  mask1

    def random_flip(self, img, mask):
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        if rand1 > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if rand2 > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask

    # def random_crop(self, img1, img2, img3, mask, size=(128, 128)):
    #     shape = img1.shape
    #     s1 = np.random.randint(0, shape[0] - size[0])
    #     s2 = np.random.randint(0, shape[1] - size[1])
    #     img1 = img1[s1:s1 + size[0], s2:s2 + size[1]]
    #     img2 = img2[s1:s1 + size[0], s2:s2 + size[1]]
    #     img3 = img3[s1:s1 + size[0], s2:s2 + size[1]]
    #     mask = mask[s1:s1 + size[0], s2:s2 + size[1]]
    #     return img1, img2, img3, mask
    #
    # def random_step(self, img1, img2, img3, mask, size=(123, 123)):
    #     shape = img1.shape
    #     s1 = np.random.randint(0, shape[0] - size[0])
    #     s2 = np.random.randint(0, shape[1] - size[1])
    #     nimg1 = np.zeros_like(img1)
    #     nimg2 = np.zeros_like(img2)
    #     nimg3 = np.zeros_like(img2)
    #     nmask = np.zeros_like(mask)
    #     img1, img2, img3, mask = self.random_crop(img1, img2, img3, mask, size)
    #     nimg1[s1: s1 + size[0], s2: s2 + size[1]] = img1
    #     nimg2[s1: s1 + size[0], s2: s2 + size[1]] = img2
    #     nimg3[s1: s1 + size[0], s2: s2 + size[1]] = img3
    #     nmask[s1: s1 + size[0], s2: s2 + size[1]] = mask
    #
    #     return nimg1, nimg2, nimg3, nmask
