# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:44
#单独分割某个类别  myo,scar,scar_edema, 通过class_index控制
"""
输入为多模态 未配准的图像，输出为单独一个前景+背景
"""

import os
import sys

# from jrs_networks.jrs_tps_seg_net import JRSTpsSegNet
from experiment.mscmr_asn_com_de import ExperimentRJ_Myo
from tools.dir import mk_or_cleardir
from tools.set_random_seed import setup_seed

import logging

#
from config.mscmr_2m_config import TpsSegNetConfigRJ_Myo

if __name__ == '__main__':

    args=TpsSegNetConfigRJ_Myo()
    setup_seed(17)
    exp= ExperimentRJ_Myo(args)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    try:
        if args.phase=='train':
            exp.train_eval_net()
        elif args.phase=='test' or args.phase=='valid':
            mk_or_cleardir(args.output_dir)
            # mk_or_cleardir(f"{args.log_dir}")
            mk_or_cleardir(f"{args.gen_dir}")
            exp.gen_res()
        elif args.phase=='metric':
            exp.metric()
        elif args.phase=="feat":
            exp.gen_feat_map()

    except KeyboardInterrupt:

        logging.info('exception interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    print("finished!!!!!!!!!")