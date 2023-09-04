from tools.np_sitk_tools import extract_label_bitwise, merge_dir
from medpy.metric import dc, asd,assd,specificity as spec, sensitivity as sens, precision as prec,ravd
from medpy.metric import hd95,hd
from tools.dir import sort_glob
import numpy as np
from tools.excel import write_array
import SimpleITK as sitk
from tools.dir import mkcleardir,mkdir_if_not_exist

import os

def cal_biasis(arrayA,arrayB,voxelspacing=None):
    if np.sum(arrayA)>np.sum(arrayB):
        return ravd(arrayA,arrayB)
    else:
        return -ravd(arrayA,arrayB)


def extract_scar(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [4]})

def extract_edema(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [2]})

def extract_scar_edema(gt_3d):
    return extract_label_bitwise(gt_3d, {1: [4]})+extract_label_bitwise(gt_3d, {1: [2]})-extract_label_bitwise(gt_3d, {1: [6]})


from tools.np_sitk_tools import reindex_label_by_dict,reindex_label_array_by_dict
'''
For PSF only
'''
def evaluation_by_dir(task_id,pre_dir,gt_dir,ref_3D,indicator=None):

    res = {}
    for it in ['scar','edema','de','t2']:
        res[it] = {'dice':[], 'hd':[],'hd95':[], 'asd':[], 'sens':[], 'prec':[],'biasis':[]}

    for subj in range(1026, 1051):
        for type in ['de','t2']:
            gts = sort_glob(f"{gt_dir}/*{subj}*")
            # gts = sort_glob(f"{gt_dir}/*{subj}*/*{type}*assn_gt_lab*")
            if type=='de':
                preds = sort_glob(f"{pre_dir}/scar/*{subj}*")
            elif type=='t2':
                preds = sort_glob(f"{pre_dir}/edema/*{subj}*")
            else:
                exit(-999)
            if len(preds)==0:
                continue
            assert len(preds) == len(gts)

            spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
            assert len(spacing_para_3D) == 1
            spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

            gt_3d = merge_dir(gts)
            pred_3d = merge_dir(preds)
            if type=='de':
                gt_3d = extract_scar(gt_3d)  # scar
                # gt_3d = reindex_label_array_by_dict(gt_3d,{1:[2221]})  # scar

            elif type=='t2':
                gt_3d = extract_edema(gt_3d)  # scar
                # gt_3d = reindex_label_array_by_dict(gt_3d,{1:[1220]})  # scar
            else:
                exit(-999)

            res[type]['dice'].append(dc(pred_3d, gt_3d))
            res[type]['sens'].append(sens(pred_3d, gt_3d))
            res[type]['prec'].append(prec(pred_3d, gt_3d))
            res[type]['hd'].append(hd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['hd95'].append(hd95(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['asd'].append(asd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['biasis'].append(cal_biasis(pred_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_id}.xls', f"{t}-{k}", res[t][k])

# def evaluation_by_dir_V2(task_id,pre_dir,gt_dir,ref_3D,indicator=None):
#
#     res = {}
#     for it in ['scar','edema','de','t2']:
#         res[it] = {'dice':[], 'hd':[],'hd95':[], 'asd':[], 'sens':[], 'prec':[]}
#
#     for subj in range(1026, 1051):
#         for type in ['de','t2']:
#             gts = sort_glob(f"{gt_dir}/*{subj}*")
#             preds = sort_glob(f"{pre_dir}/*{subj}*")
#             assert len(preds) == len(gts)
#
#             spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
#             assert len(spacing_para_3D)==1
#             spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()
#
#             gt_3d = merge_dir(gts)
#             scar_3d = merge_dir(preds)
#             if type=='de':
#                 gt_3d_scar = extract_scar(gt_3d)  # scar
#             elif type=='t2':
#                 gt_3d_scar = extract_scar(gt_3d)  # scar
#
#             res[type]['dice'].append(dc(scar_3d, gt_3d_scar))
#             res[type]['sens'].append(sens(scar_3d, gt_3d_scar))
#             res[type]['prec'].append(prec(scar_3d, gt_3d_scar))
#             res[type]['hd'].append(hd(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#             res[type]['hd95'].append(hd95(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#             res[type]['asd'].append(asd(scar_3d, gt_3d_scar, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
#
#     for t in res.keys():
#         print(f"===={t}=======")
#         for k in res[t].keys():
#             # print(res[t][k])
#             if len(res[t][k]) <= 0:
#                 continue
#             print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
#             write_array(f'../outputs/result/{task_id}.xls', f"{t}-{k}", res[t][k])


'''
For nnunet only
'''
def evaluation_by_dir_SM(task_name, input_dir, gt_dir, ref_3D, type):
    # 网络输出的数据

    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[]}

    for subj in range(2021, 2050):

        preds = sort_glob(f"{input_dir}/subject*{subj}*")
        gts = sort_glob(f"{gt_dir}/subject*{subj}*")
        assert len(preds) == len(gts)
        print(len(gts))
        gt_3d = merge_dir(gts)
        preds_3d = merge_dir(preds)
        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
        spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])

'''
For nnunet only
'''
def evaluation_by_dir_awsnet(task_name, input_dir, gt_dir, ref_3D, type):
    # 网络输出的数据

    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[]}

    for subj in range(2021, 2050):

        preds = sort_glob(f"{input_dir}/subject*{subj}*")
        gts = sort_glob(f"{gt_dir}/subject*{subj}*")
        assert len(preds) == len(gts)
        print(len(gts))
        gt_3d = merge_dir(gts)
        preds_3d = merge_dir(preds)

        gt_3d=reindex_label_array_by_dict(gt_3d,{1:[2]})
        preds_3d=reindex_label_array_by_dict(preds_3d,{1:[2]})
        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*{type}.nii.gz")
        spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])



'''
For nnunet only
'''
def evaluation_by_dir_ants_seg(task_name, input_dir, gt_dir, ref_3D, type):
    # 网络输出的数据

    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[]}

    for subj in range(2021, 2050):

        preds = sort_glob(f"{input_dir}/*{subj}*")
        gts = sort_glob(f"{gt_dir}/*{subj}*")
        assert len(preds) == len(gts)
        print(len(gts))
        gt_3d = merge_dir(gts)
        preds_3d = merge_dir(preds)

        if type=='scar':
            gt_3d=reindex_label_array_by_dict(gt_3d,{1:[2]})
            preds_3d=reindex_label_array_by_dict(preds_3d,{1:[2]})
        else:
            gt_3d = reindex_label_array_by_dict(gt_3d, {1: [1,2]})
            preds_3d = reindex_label_array_by_dict(preds_3d, {1: [1,2]})

        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*.nii.gz")
        spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])




from visulize.color import colorGD,colorC0,colorT2,colorDe


'''
For PSF only
'''
def evaluation_by_dir_joint(task_id,pre_dir,gt_dir,ref_3D):

    res = {}
    for it in ['scar','edema']:
        res[it] = {'dice':[], 'hd':[],'hd95':[], 'asd':[], 'sens':[], 'prec':[],'biasis':[]}
    print(res)
    for subj in range(2021, 2050):

        for type in ['scar','edema']:
            gts = sort_glob(f"{gt_dir}/*{subj}*")
            preds=sort_glob(f"{pre_dir}/*{subj}*")
            assert len(preds) == len(gts)
            print((preds))
            print((gts))
            gt_3d = merge_dir(gts)
            pred_3d = merge_dir(preds)
            if type=='scar':
                spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*[de|DE].nii.gz")
                assert len(spacing_para_3D) == 1
                spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

                gt_3d = reindex_label_array_by_dict(gt_3d,{2:[2]})  # scar
                pred_3d = reindex_label_array_by_dict(pred_3d,{2:[2]})  # scar

            elif type=='edema':
                spacing_para_3D = sort_glob(f"{ref_3D}/*{subj}*[t2|T2].nii.gz")
                assert len(spacing_para_3D) == 1
                spacing_para = sitk.ReadImage(spacing_para_3D[0]).GetSpacing()

                gt_3d = reindex_label_array_by_dict(gt_3d,{1:[1,2]})  # scar
                pred_3d = reindex_label_array_by_dict(pred_3d,{1:[1,2]})  # scar

            else:
                exit(-999)
            print(f"{subj}")
            res[type]['dice'].append(dc(pred_3d, gt_3d))
            res[type]['sens'].append(sens(pred_3d, gt_3d))
            res[type]['prec'].append(prec(pred_3d, gt_3d))
            res[type]['hd'].append(hd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['hd95'].append(hd95(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['asd'].append(asd(pred_3d, gt_3d, voxelspacing=(spacing_para[-1], spacing_para[1], spacing_para[0])))
            res[type]['biasis'].append(cal_biasis(pred_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    print(res)
    path=f'../outputs/result/{task_id}.xls'
    if os.path.exists(path):
        os.remove(path)

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(path, f"{t}-{k}", res[t][k])

def evaluation_reg_by_dir_joint(gd_paths, pred_paths,ref_3D, roi_labs={1: [1220, 2221, 200, 500]}):
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
        preds_arr = np.squeeze(reindex_label_array_by_dict(preds_arr, roi_labs))
        preds_arr = np.squeeze(preds_arr)
        ds_res = dc(gds_arr, preds_arr)
        if len(gds_arr.shape) == 2:
            gds_arr = np.expand_dims(gds_arr, axis=-1)
            preds_arr = np.expand_dims(preds_arr, axis=-1)

        suject = "_".join(os.path.basename(gd_paths[0]).split('.')[0].split("_")[:-1])

        para = sitk.ReadImage(ref_3D)
        # hds[modality].append(hd95(gds_arr,preds_arr,para.GetSpacing()))

        hd_res = hd(gds_arr, preds_arr, (para.GetSpacing()[-1], para.GetSpacing()[1], para.GetSpacing()[0]))
        asd_res = asd(gds_arr, preds_arr, (para.GetSpacing()[-1], para.GetSpacing()[1], para.GetSpacing()[0]))
        return ds_res, hd_res, asd_res


def myo_infarct_size(task_name, pred_dir, gt_dir, ref_3D, type, interst={1:[2221]}):
    # 网络输出的数据

    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[],'infarct_size':[]}

    for one_pred_dir,one_gt_dir,one_ref_path in zip(pred_dir, gt_dir, ref_3D):

        print(f"{one_pred_dir}  {one_gt_dir}")
        # assert len(inputdir) == len(gtdir)
        pathes_gt=sort_glob(f"{one_gt_dir}/*{type}_assn_gt_lab*")
        pathes_pred= sort_glob(f"{one_pred_dir}/*{type}_assn_gt_lab*")
        assert len(pathes_pred)==len(pathes_pred)
        gt_3d = merge_dir(pathes_gt)
        preds_3d = merge_dir(pathes_pred)

        myo_gt_3d=reindex_label_array_by_dict(gt_3d,{1:[2221,200,1220]})
        gt_3d=reindex_label_array_by_dict(gt_3d,{1:[2221]})

        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para = sitk.ReadImage(one_ref_path).GetSpacing()

        res[type]['infarct_size'].append(np.sum(preds_3d)/np.sum(myo_gt_3d))

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        # res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        # res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        # res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])



def myo_edema_size(task_name, pred_dir, gt_dir, ref_3D, type, interst={1:[2221]}):
    # 网络输出的数据

    res = {}
    for it in ['scar', 'edema', 'de', 't2','DE',"T2"]:
        res[it] = {'dice': [], 'hd': [], 'hd95': [], 'asd': [], 'sens': [], 'prec': [],'biasis':[],'infarct_size':[]}

    for one_pred_dir,one_gt_dir,one_ref_path in zip(pred_dir, gt_dir, ref_3D):

        print(f"{one_pred_dir}  {one_gt_dir}")
        # assert len(inputdir) == len(gtdir)
        pathes_gt=sort_glob(f"{one_gt_dir}/*{type}_assn_gt_lab*")
        pathes_pred= sort_glob(f"{one_pred_dir}/*{type}_assn_gt_lab*")
        assert len(pathes_pred)==len(pathes_pred)
        gt_3d = merge_dir(pathes_gt)
        preds_3d = merge_dir(pathes_pred)

        myo_gt_3d=reindex_label_array_by_dict(gt_3d,{1:[2221,200,1220]})
        gt_3d=reindex_label_array_by_dict(gt_3d,{1:[1220]})

        #https://blog.csdn.net/tangxianyu/article/details/102454611
        spacing_para = sitk.ReadImage(one_ref_path).GetSpacing()

        # if np.sum(preds_3d)==0:
        #     continue

        res[type]['infarct_size'].append(np.sum(preds_3d)/np.sum(myo_gt_3d))

        res[type]['dice'].append(dc(preds_3d, gt_3d))
        res[type]['sens'].append(sens(preds_3d, gt_3d))
        res[type]['prec'].append(prec(preds_3d, gt_3d))
        # res[type]['asd'].append(asd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        # res[type]['hd'].append(hd(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        # res[type]['hd95'].append(hd95(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))
        res[type]['biasis'].append(cal_biasis(preds_3d, gt_3d,voxelspacing=(spacing_para[-1],spacing_para[1],spacing_para[0])))

    for t in res.keys():
        print(f"===={t}=======")
        for k in res[t].keys():
            # print(res[t][k])
            if len(res[t][k]) <= 0:
                continue
            print(f"mean {t}-{k}:{np.mean(res[t][k])} {np.std(res[t][k])}")
            write_array(f'../outputs/result/{task_name}.xls', f"{t}-{k}", res[t][k])
