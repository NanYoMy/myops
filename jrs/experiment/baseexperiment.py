import matplotlib.pyplot
from torch.utils.tensorboard  import SummaryWriter
import torch

from tools.dir import sort_glob
from medpy.metric import dc,assd,asd
# from medpy.metric import hd95 as hd
from medpy.metric import hd

import logging
import cv2
from tools.np_sitk_tools import reindex_label_array_by_dict

from tools.np_sitk_tools import clipseScaleSitkImage,clipseScaleSArray
from skimage.util.compare import compare_images
from skimage.exposure import rescale_intensity
from tools.excel import write_array
from visulize.color import my_color
from numpy import ndarray
import cv2
from tools.dir import mkdir_if_not_exist
import os
import numpy as np
from tools.set_random_seed import setup_seed
from tools.itkdatawriter import sitk_write_image,sitk_write_images,sitk_write_labs,sitk_write_lab
from medpy.metric import dc,asd,assd
from tools.np_sitk_tools import sitkResize3D
from baseclass.medicalimage import Modality,MyoPSLabelIndex
import SimpleITK as sitk
from skimage import  segmentation,color
from matplotlib.pyplot import plot
import seaborn
from skimage import  io,measure,morphology
class BaseExperiment():
    def write_dict_to_tb(self,dict,step):
        for k in dict.keys():
            self.eval_writer.add_scalar(f"{k}",dict[k],step)

    def __init__(self,args):
        self.args=args
        self.eval_writer = SummaryWriter(log_dir=f"{args.log_dir}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
class BaseMyoPSExperiment(BaseExperiment):
    def get_target_size(self,array_size):
        w,h=array_size

        # new_h=500
        # new_w = int(500/h*w)
        # return (new_h,new_w)
        new_x=500
        new_y = int(500/h*w)
        return (new_x,new_y)


    def save_tensor_with_parameter(self, tensor, parameter, outputdir, name, is_label=False):
        array=tensor.cpu().numpy()
        array=np.squeeze(array)
        target_size=parameter.GetSize()
        if is_label==True:
            array=self.op.resize(array,(target_size[1],target_size[0]),0)
        else:
            array=self.op.resize(array,(target_size[1],target_size[0]),1)

        array=np.expand_dims(array,axis=0)
        if is_label==True:
            array=np.round(array).astype(np.int16)

        img = sitk.GetImageFromArray(array)
        img.CopyInformation(parameter)
        sitk.WriteImage(img, os.path.join(outputdir, name+'.nii.gz'))



    def create_torch_tensor(self, img_c0, img_de, img_t2, lab_c0, lab_de, lab_t2):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_scar_mask, scar_mask, lv_mask, rv_mask,ori_label
        c0_roi_reg_mask_1 = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_1 = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_1 = lab_de.narrow(dim=1, start=1, length=1)
        c0_roi_reg_mask_2 = lab_c0.narrow(dim=1, start=4, length=1)
        t2_roi_reg_mask_2 = lab_t2.narrow(dim=1, start=4, length=1)
        de_roi_reg_mask_2 = lab_de.narrow(dim=1, start=4, length=1)
        c0_roi_reg_mask_3 = lab_c0.narrow(dim=1, start=5, length=1)#rv
        t2_roi_reg_mask_3 = lab_t2.narrow(dim=1, start=5, length=1)#rv
        de_roi_reg_mask_3 = lab_de.narrow(dim=1, start=5, length=1)#rv
        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}
        roi_lab1={Modality.c0:c0_roi_reg_mask_1, Modality.t2:t2_roi_reg_mask_1, Modality.de:de_roi_reg_mask_1}
        roi_lab2={Modality.c0:c0_roi_reg_mask_2, Modality.t2:t2_roi_reg_mask_2, Modality.de:de_roi_reg_mask_2}
        roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab1,roi_lab2

    def __init__(self,args):
        super().__init__(args)


from PIL import Image, ImageOps
class BaseMSCMRExperiment(BaseMyoPSExperiment):
    def create_torch_tensor(self, img_c0, img_t2,img_de,  lab_c0, lab_t2, lab_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        lab_c0 = lab_c0.to(device=self.device, dtype=torch.float32)
        lab_t2 = lab_t2.to(device=self.device, dtype=torch.float32)
        lab_de = lab_de.to(device=self.device, dtype=torch.float32)
        #bg_mask, myo_mask, edema_mask, scar_mask, lv_mask, rv_mask,myo_scar_mask,myo_ede_mask,myo_scar_ede_mask,ori_lab
        c0_roi_reg_mask_myo = lab_c0.narrow(dim=1, start=1, length=1)
        t2_roi_reg_mask_myo = lab_t2.narrow(dim=1, start=1, length=1)
        de_roi_reg_mask_myo = lab_de.narrow(dim=1, start=1, length=1)

        c0_roi_reg_mask_lv = lab_c0.narrow(dim=1, start=4, length=1)
        t2_roi_reg_mask_lv = lab_t2.narrow(dim=1, start=4, length=1)
        de_roi_reg_mask_lv = lab_de.narrow(dim=1, start=4, length=1)

        c0_roi_reg_mask_rv = lab_c0.narrow(dim=1, start=5, length=1)
        t2_roi_reg_mask_rv = lab_t2.narrow(dim=1, start=5, length=1)
        de_roi_reg_mask_rv = lab_de.narrow(dim=1, start=5, length=1)


        lab_c0 = lab_c0.narrow(dim=1, start=-1, length=1)
        lab_t2 = lab_t2.narrow(dim=1, start=-1, length=1)
        lab_de = lab_de.narrow(dim=1, start=-1, length=1)

        img={Modality.c0:img_c0, Modality.t2:img_t2, Modality.de:img_de}
        lab={Modality.c0:lab_c0, Modality.t2:lab_t2, Modality.de:lab_de}

        roi_lab_myo={Modality.c0:c0_roi_reg_mask_myo, Modality.t2:t2_roi_reg_mask_myo, Modality.de:de_roi_reg_mask_myo}
        roi_lab_lv={Modality.c0:c0_roi_reg_mask_lv, Modality.t2:t2_roi_reg_mask_lv, Modality.de:de_roi_reg_mask_lv}
        roi_lab_rv={Modality.c0:c0_roi_reg_mask_rv, Modality.t2:t2_roi_reg_mask_rv, Modality.de:de_roi_reg_mask_rv}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return  img,lab,roi_lab_myo,roi_lab_lv,roi_lab_rv

    def create_test_torch_tensor(self, img_c0, img_t2, img_de):
        img_c0 = img_c0.to(device=self.device, dtype=torch.float32)  # *C0_lv_pool_gd
        img_t2 = img_t2.to(device=self.device, dtype=torch.float32)  # *T2_lv_pool_gd
        img_de = img_de.to(device=self.device, dtype=torch.float32)  # *DE_lv_pool_gd
        img = {Modality.c0: img_c0, Modality.t2: img_t2, Modality.de: img_de}
        # roi_lab3={Modality.c0:c0_roi_reg_mask_3, Modality.t2:t2_roi_reg_mask_3, Modality.de:de_roi_reg_mask_3}
        return img


    def renamepath(self, name, tag):
        term = os.path.basename((name[0])).split("_")
        name=f'{term[0]}_{term[1]}_{term[2]}_{term[3]}_{tag}_{term[4]}'
        return name





    def save_tensor_with_parameter(self, tensor, parameter, outputdir, name, is_label=False):

        if not isinstance(tensor,ndarray):
            array=tensor.cpu().numpy()
        else:
            array=tensor
        array=np.squeeze(array)
        target_size=parameter.GetSize()
        if is_label==True:
            array=self.op.resize(array,(target_size[1],target_size[0]),0)
        else:
            array=self.op.resize(array,(target_size[1],target_size[0]))

        array=np.expand_dims(array,axis=0)
        if is_label==True:
            array=np.round(array).astype(np.int16)

        img = sitk.GetImageFromArray(array)
        img.CopyInformation(parameter)
        sitk.WriteImage(img, os.path.join(outputdir, name+'.nii.gz'))


    def print_res(self, seg_ds, seg_hds,task='seg'):
        for k in seg_ds.keys():
            if (len(seg_ds[k])) > 0:
                # print(ds[k])
                logging.info(f'subject level evaluation:  DS {k}: {np.mean(seg_ds[k])} {np.std(seg_ds[k])}')
                write_array(self.args.res_excel, f'myops_asn_{task}_{k}_ds', seg_ds[k])
                # print(hds[k])
                logging.info(f'subject level evaluation:  HD {k}: {np.mean(seg_hds[k])} {np.std(seg_hds[k])}')
                write_array(self.args.res_excel, f'myops_asn_{task}_{k}_hd', seg_hds[k])

    def cal_ds_hd(self, gd_paths, pred_paths, roi_labs={1: [1220, 2221, 200, 500]}):
        seg_gds_list = []
        seg_preds_list = []
        assert len(gd_paths) == len(pred_paths)
        for gd, pred in zip(gd_paths, pred_paths):
            # print(f"{gd}-{pred}")
            seg_gds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gd)), axis=0))
            seg_preds_list.append(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(pred)), axis=0))
        gds_arr = np.concatenate(seg_gds_list, axis=0)
        preds_arr = np.concatenate(seg_preds_list, axis=0)
        gds_arr = np.squeeze(reindex_label_array_by_dict(gds_arr, roi_labs))
        preds_arr = np.squeeze(preds_arr)
        ds_res = dc(gds_arr, preds_arr)
        if len(gds_arr.shape) == 2:
            gds_arr = np.expand_dims(gds_arr, axis=-1)
            preds_arr = np.expand_dims(preds_arr, axis=-1)


        suject="_".join(os.path.basename(gd_paths[0]).split('.')[0].split("_")[:-1])

        gd3d=sort_glob(f"../data/gen_{self.args.data_source}/croped/{suject}.nii.gz")
        para=sitk.ReadImage(gd3d[0])

        # labels=measure.label(np.squeeze(preds_arr))
        # preds_arr = morphology.remove_small_objects(labels, min_size=100)

        hd_res = hd(np.squeeze(gds_arr), np.squeeze(preds_arr), (1,1))
        # asd_res = asd(gds_arr, preds_arr, (para.GetSpacing()[-1],para.GetSpacing()[1],para.GetSpacing()[0]))
        asd_res = asd(np.squeeze(gds_arr), np.squeeze(preds_arr), (1,1))
        return ds_res, hd_res,asd_res
    def __init__(self,args):
        super().__init__(args)