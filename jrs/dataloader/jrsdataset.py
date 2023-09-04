#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
这个文件用于读取mscmr比赛的数据集合，
'''
import os
from torch.utils import data
import numpy as np
import torch
import random
# def data_augmentation(self, img1, img2, img3, mask):
#     #
#     # img1, img2, img3, mask = self.random_rotate(img1, img2, img3, mask)
#     # img1, img2, img3, mask = self.random_flip(img1, img2, img3, mask)
#     # img1, img2, img3, mask = self.random_step(img1, img2, img3, mask)
#     rd_scale = np.random.uniform(0.9, 1.2)
#     rd_translate_x = np.random.uniform(-0.1, 0.1) * img1.shape[0]
#     rd_translate_y = np.random.uniform(-0.1, 0.1) * img1.shape[1]
#     rd_rotate = np.random.uniform(-np.pi / 6, np.pi / 6)
#
#     transform.AffineTransform(scale=rd_scale, translation=(rd_translate_x, rd_translate_y), rotation=rd_rotate)
#
#     img1 = self.masked_normalize(img1.astype("float"), mask)
#     img2 = self.masked_normalize(img2.astype("float"), mask)
#     img3 = self.masked_normalize(img3.astype("float"), mask)
#     return img1, img2, img3, mask
from baseclass.medicalimage import MyoPSLabelIndex
from dataloader.util import make_numpy_one_hot, SkimageOP_Base, SkimageOP_jrs_Pathology, SKimage_Single_modality
from tools.dir import sort_glob


def myops_dataset(dirname, filter="C0"):
    result = []  # 含有filter的所有的文件
    for maindir, subdir, file_name_list in os.walk(dirname):

        # print("1:",maindir) #当前主目录
        # print("2:",subdir) #当前主目录下的所有目录
        # print("3:",file_name_list)  #当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            # ext = os.path.splitext(apath)[1]  # 获取文件后缀 [0]获取的是除了文件名以外的内容
            ext = apath.split("_")[-2]
            if ext in filter:
                result.append(apath)
    return result
import SimpleITK as sitk
def load_slicer(path):
    C0_path = path
    DE_path = C0_path.replace("C0", "DE")
    T2_path = C0_path.replace("C0", "T2")
    c0_gdpath = C0_path.replace("C0", "C0_gd")
    t2_gdpath = C0_path.replace("C0", "T2_gd")
    de_gdpath = C0_path.replace("C0", "DE_gd")


    # p, gdname = os.path.split(gdpath);
    # preadname = gdname.replace("gd", "pred")
    # img_C0 = nib.load(C0_path).get_data()
    # img_DE = nib.load(DE_path).get_data()
    # img_T2 = nib.load(T2_path).get_data()
    # img_gd = nib.load(gdpath).get_data()
    img_C0 =sitk.GetArrayFromImage(sitk.ReadImage(C0_path)).astype(np.float)
    img_DE =sitk.GetArrayFromImage(sitk.ReadImage(DE_path)).astype(np.float)
    img_T2 =sitk.GetArrayFromImage(sitk.ReadImage(T2_path)).astype(np.float)
    C0_gd =sitk.GetArrayFromImage(sitk.ReadImage(c0_gdpath)).astype(np.float)
    LGE_gd =sitk.GetArrayFromImage(sitk.ReadImage(de_gdpath)).astype(np.float)
    T2_gd =sitk.GetArrayFromImage(sitk.ReadImage(t2_gdpath)).astype(np.float)

    return img_C0, img_DE, img_T2, C0_gd,LGE_gd,T2_gd#, preadname, gdname

'''
this dataloader returns bg,myo,edema,scar
'''

class DataSet_unliagned(data.Dataset):
    def __init__(self, args, type="train", augo=True, task="myo",ret_name=False):
        self.args = args
        self.augo = augo
        self.task = task
        self.op = SkimageOP_Base()
        self.SIZE=(256,256) # output image size

        subjects=sort_glob(args.dataset_dir+"/*")

        if type == "train":
            self.train = True
            subjects=subjects[:20]

        elif type== "test":
            self.train = False
            subjects=subjects[20:]
        elif type=='all':
            self.train = False
            subjects = subjects
        else:
            self.train = False
            subjects=subjects

        # print(f"{__class__.__name__}")
        print(subjects)

        self.c0=self._getsample(subjects, "c0")
        self.t2=self._getsample(subjects, "t2")
        self.de=self._getsample(subjects, "de")
        self.cur_index=-1
        self.ret_name=ret_name
    def get_cur_path(self):
        return self.c0[self.cur_index],self.t2[self.cur_index],self.de[self.cur_index]

    def __getitem__(self, index):
        self.cur_index=index
        # path
        # cur_path = self.data_paths[index]
        # print(f'{cur_path}')
        # get images
        # img_t1,img_t1ce,img_t2,img_flair = loadSubjectData(cur_path)
        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)
        # 在数据预处理的时候就进行归一化处理
        # mysize=(self.args.image_size,self.args.image_size,)
        # img_c0,lab_c0 = self.op.normalize_image(img_c0.astype("float"),lab_c0.astype("float"),size=mysize,clip=True) #对于结构的配准与分割可以考虑clip
        # img_t2,lab_t2 = self.op.normalize_image(img_t2.astype("float"),lab_t2.astype("float"),size=mysize,clip=True)
        # img_de,lab_de = self.op.normalize_image(img_de.astype("float"),lab_de.astype("float"),size=mysize,clip=True)

        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseq(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if not self.ret_name:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["lab"][index],self.t2["lab"][index],self.de["lab"][index]

    def _create_label(self, gd_lab):
        ori_label = gd_lab[np.newaxis, :, :]
        ori_label = np.round(ori_label).astype('uint16')

        # 只有myo的信息，并且去除其他非相关的标签 0:背景 1:myo 3:scare 2:edema 4:其他
        myo_mask = np.zeros_like(ori_label)
        myo_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.scar.value] = 1
        myo_mask[ori_label == MyoPSLabelIndex.edema.value] = 1

        # 只有edema的信息
        edema_mask = np.zeros_like(ori_label)
        edema_mask[ori_label == MyoPSLabelIndex.edema.value] = 1

        # 只有scar的信息
        scar_mask = np.zeros_like(ori_label)
        scar_mask[ori_label == MyoPSLabelIndex.scar.value] = 1

        # 只有lv_p的信息
        lv_pool_mask = np.zeros_like(ori_label)
        lv_pool_mask[ori_label == MyoPSLabelIndex.lv_p.value] = 1
        lv_mask=lv_pool_mask

        # 只有rv的信息
        rv_mask = np.zeros_like(ori_label)
        rv_mask[ori_label == MyoPSLabelIndex.rv.value] = 1

        bg_mask = np.where(ori_label == 0, 1, 0)


        # scar 与 myo
        myo_scar_mask = np.zeros_like(ori_label)
        myo_scar_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.edema.value] = 1
        myo_scar_mask[ori_label == MyoPSLabelIndex.scar.value] = 2

        #scar 与 myo
        myo_ede_mask = np.zeros_like(ori_label)
        myo_ede_mask[ori_label == MyoPSLabelIndex.myo.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.scar.value] = 1
        myo_ede_mask[ori_label == MyoPSLabelIndex.edema.value] = 2



        mmc_label = np.concatenate([bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask], axis=0)

        mmc_label = torch.from_numpy(mmc_label).float()

        return mmc_label

    def __len__(self):
        assert len(self.c0["img"])==len(self.t2["img"])
        assert len(self.c0["img"])==len(self.de["img"])
        return len(self.c0["img"])

    def _getsample(self, subjects, type):
        dict = {"img": [], "lab": []}
        for s in subjects:
            dict["img"].extend(sort_glob(f"{s}/*img_{type}*"))
            if type=="c0":
                dict["lab"].extend(sort_glob(f"{s}/*ana_{type}*"))
            else:
                dict["lab"].extend(sort_glob(f"{s}/*ana_patho_{type}*"))
        return dict



    def _readimg(self,index):
        # print(self.t2["img"][index])
        img_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["img"][index])).astype(np.float)
        img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["img"][index])).astype(np.float)
        img_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["img"][index])).astype(np.float)

        lab_c0 = sitk.GetArrayFromImage(sitk.ReadImage(self.c0["lab"][index])).astype(np.float)
        lab_t2 = sitk.GetArrayFromImage(sitk.ReadImage(self.t2["lab"][index])).astype(np.float)
        lab_de = sitk.GetArrayFromImage(sitk.ReadImage(self.de["lab"][index])).astype(np.float)

        print("============================")
        self.check(index)
        print("============================")


        return img_c0,  img_t2, img_de, lab_c0,lab_t2, lab_de

    def check(self, index):
        img_terms = os.path.basename(self.de['img'][index]).split('_')
        lab_terms = os.path.basename(self.de['lab'][index]).split('_')
        assert img_terms[1] == lab_terms[1]
        assert img_terms[-1] == lab_terms[-1]




'''
large and small spatial augmentation
'''
class DataSetRJ(DataSet_unliagned):
    def __getitem__(self, index):
        self.cur_index=index

        # print(f"current {self.cur_index}")
        img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de = self._readimg(index)

        img_c0, lab_c0 = self.op.normalize_image_label(img_c0, lab_c0, (256, 256), True)
        img_t2, lab_t2 = self.op.normalize_image_label(img_t2, lab_t2, (256, 256), True)
        img_de, lab_de = self.op.normalize_image_label(img_de, lab_de, (256, 256), True)
        img_c0 = self.op.usm(img_c0)
        img_t2 = self.op.usm(img_t2)
        img_de = self.op.usm(img_de)


        if self.augo:
            img_c0, img_t2, img_de, lab_c0, lab_t2, lab_de = self.op.aug_multiseqV3(img_c0, img_t2, img_de, lab_c0,lab_t2, lab_de,0.2)
            img_c0 = self.op.gamma_correction(img_c0)
            img_de = self.op.gamma_correction(img_de)
            img_t2 = self.op.gamma_correction(img_t2)

        c0_mmc_gd = self._create_label(lab_c0)
        t2_mmc_gd = self._create_label(lab_t2)
        lge_mmc_gd = self._create_label(lab_de)

        img_c0 = self.op.convert_array_2_torch(img_c0)
        img_t2 = self.op.convert_array_2_torch(img_t2)
        img_de = self.op.convert_array_2_torch(img_de)

        if not self.ret_name:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd
        else:
            return img_c0, img_t2, img_de, c0_mmc_gd, t2_mmc_gd, lge_mmc_gd,self.c0["img"][index],self.t2["img"][index],self.de["img"][index]



