



import logging
import argparse




class Tps2MSegNetConfigRJ_myo():
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")
        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='RJ_unaligned,myops20')
        parser.add_argument('--modalities', dest='modalities', default='c0,de', help='de,c0   t2,c0    de,t2')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)

        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range
        self.modalities=self.cmd_args.modalities

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config

        self.model_id = f"2M_asn_{args.modalities}_myo_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)

class Tps2MSegNetConfigRJ_myo_WO_STF():
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")
        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='RJ_unaligned,myops20')
        parser.add_argument('--modalities', dest='modalities', default='de,c0', help='de,c0   t2,c0    de,t2')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)

        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range
        self.modalities=self.cmd_args.modalities

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config

        self.model_id = f"2M_asn_wo_STF_{args.modalities}_myo_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)


class MSCMRPahtologySegConfig():
    def __init__(self):

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='# train iteration')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,
                            help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,
                            help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000],
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,
                            help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,
                            help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True,
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50,
                            help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4,
                            help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1,
                            help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight', type=float, dest='weight', default=1, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200',
                            help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False,
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False,
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,
                            help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,
                            help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='test', help='train,test,gen')
        parser.add_argument('--dataset_dir', dest='dataset_dir', default='../outputs/jrs_2_tps_unaligned_4/gen_res',help='')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')
        parser.add_argument('--backbone', nargs="+", default=[4,8,16,32,64], type=int)
        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        # static config
        self.epochs = self.cmd_args.epochs
        self.init_epochs = self.cmd_args.init_epochs
        self.n_label = self.cmd_args.n_label
        self.image_size = self.cmd_args.image_size
        self.momentum = self.cmd_args.momentum
        self.lr_decay_gamma = self.cmd_args.lr_decay_gamma
        self.weight_decay = self.cmd_args.weight_decay
        self.weight = self.cmd_args.weight
        self.nesterov = self.cmd_args.nesterov
        self.lr_decay_milestones = self.cmd_args.lr_decay_milestones
        self.optimizer = self.cmd_args.optimizer
        self.batch_size = self.cmd_args.batch_size
        self.gen_num = self.cmd_args.gen_num
        self.decay_freq = self.cmd_args.decay_freq
        self.save_freq = self.cmd_args.save_freq
        self.print_freq = self.cmd_args.print_freq
        self.save_cp = self.cmd_args.save_cp
        # self.cc_feat = self.cmd_args.ccfeat
        self.max_size = self.cmd_args.max_size
        self.num_channel_initial = self.cmd_args.num_channel_initial
        self.group_num = self.cmd_args.group_num
        self.nb_class = self.cmd_args.nb_class
        self.class_index = self.cmd_args.class_index
        self.print_tb = self.cmd_args.print_tb
        self.fold = self.cmd_args.fold
        self.scale = self.cmd_args.scale
        self.components = self.cmd_args.components
        self.task = self.cmd_args.task
        self.val_percent = self.cmd_args.val_percent
        self.out_threshold = self.cmd_args.out_threshold
        self.load = self.cmd_args.load
        self.save_imgs = self.cmd_args.save_imgs
        self.phase = self.cmd_args.phase
        # self.data_source = self.cmd_args.data_source
        self.lr = self.cmd_args.lr
        self.net = self.cmd_args.net
        self.backbone=self.cmd_args.backbone
        # self.grid_size = self.cmd_args.grid_size
        # self.span_range = self.cmd_args.span_range

        if self.phase == 'test' or self.phase == 'gen':
            self.load = True
        # self.span_range_height = self.span_range_width = self.span_range
        # self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)

    def dynamic_config(self, args):
        # dynamica config
        #we need the original
        # self.dataset_dir = '../outputs/gen_res/' % (self.model_id)
        #set new model id
        self.model_id = f"patho_2_seg_backbone_{'x'.join([str(i) for i in self.backbone])}"
        self.dataset_dir=args.dataset_dir
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'
        self.output_dir = '../outputs/%s/res' % (self.model_id)


class TpsSegNetConfigRJ(Tps2MSegNetConfigRJ_myo):
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight', type=float, dest='weight', default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='RJ_unaligned', help='RJ_unaligned,myops20')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config
        self.model_id = f"asn_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)


class TpsSegNetConfigRJ_Myo():
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='ZS_unaligned,myops20')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize'or self.phase=='feat':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config

        self.model_id = f"asn_myo_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)


class TpsSegMvMMNetConfigRJ_Myo():
    def __init__(self):
        parser = argparse.ArgumentParser(description='None')
        parser.add_argument('--epochs', dest='epochs', type=int, default=5000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--info', dest='info', default=None, help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='ZS_unaligned,myops20')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.00001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range


        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize'or self.phase=='feat':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config
        if self.cmd_args.info!=None:
            self.model_id=self.cmd_args.info
        else:
            self.model_id = f"asn_myo_mvmmloss_tps_{args.net}_{args.data_source}_{args.weight}"

        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)


class TpsSegNetConfigRJ_Myo_wo_STF():
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='RJ_unaligned,myops20')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize'or self.phase=='feat':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config

        self.model_id = f"wo_STF_myo_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)


class TpsSegNetConfigRJ_Myo_wo_SegConstraint():
    def __init__(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epochs', dest='epochs', type=int, default=2000, help='# train iteration')
        parser.add_argument('--ckpt', dest='ckpt', type=int, default=-1, help='# check point for test')
        parser.add_argument('--init_epochs', dest='init_epochs', type=int, default=0, help='# train iteration')
        parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
        parser.add_argument('--image_size', dest='image_size', type=int, default=256, help='the size of image_size')
        parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.9,   help='initial learning rate for adam')
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-8,   help='initial learning rate for adam')
        parser.add_argument('--nesterov', dest='nesterov', type=bool, default=True,      help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--lr_decay_milestones', dest='lr_decay_milestones', nargs='*', default=[20, 2000], help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--optimizer', dest='optimizer', default='Adam', help='')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
        parser.add_argument("--gen_num", dest='gen_num', type=int, nargs=1, default=3000, help="")
        parser.add_argument('--decay_freq', dest='decay_freq', type=int, default=1000, help='decay frequent')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=50,   help='save a model every save_freq iterations')
        parser.add_argument('--print_freq', dest='print_freq', type=int, default=500,  help='print the debug information every print_freq iterations')
        parser.add_argument('--save_cp', dest='save_cp', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
        parser.add_argument('--group_num', dest='group_num', type=int, default=2, help='miccai:32,')
        parser.add_argument('--nb_class', dest='nb_class', type=int, default=1, help='miccai:32,')
        parser.add_argument('--class_index', dest='class_index', type=int, default=1, help='myo:1 scaredema:2 scar:3')
        parser.add_argument('--weight',  dest='weight',type=float, default=1.0, help='weight of reg loss ')
        parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')
        parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--components', dest='components', type=str, default='2220-2221-200', help='205=myocardium 500=lv')
        parser.add_argument('--task', dest='task', default='myops', help='myops')
        parser.add_argument('--val_percent', dest='val_percent', type=float, default=0.25, help='')
        parser.add_argument('--out_threshold', dest='out_threshold', type=float, default=0.5, help="threshold ")

        parser.add_argument('--load', dest='load', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--print_tb', dest='print_tb', action='store_true', default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--save_imgs', dest='save_imgs', action='store_true', default=False,  help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--ccfeat', dest='ccfeat', action='store_true', default=True,  help='if continue training, load the latest model: 1: true, 0: false')

        parser.add_argument('--phase', dest='phase', default='metric', help='train,test,gen')
        parser.add_argument('--data_source', dest='data_source', default='ZS_unaligned', help='RJ_unaligned,myops20')
        parser.add_argument('--log_level', dest='log_level', default=10, help='10 debug, 20 info, error 40')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--net', dest='net', default='tps', help='tps,aff')
        # tps related config
        parser.add_argument('--grid_size', dest='grid_size', type=int, default=4, help='default is 4')
        parser.add_argument('--span_range', dest='span_range', type=int, default=0.98)
        self.cmd_args = parser.parse_args()
        #static config
        self.epochs=self.cmd_args.epochs
        self.ckpt=self.cmd_args.ckpt
        self.init_epochs=self.cmd_args.init_epochs
        self.n_label=self.cmd_args.n_label
        self.image_size=self.cmd_args.image_size
        self.momentum=self.cmd_args.momentum
        self.lr_decay_gamma=self.cmd_args.lr_decay_gamma
        self.weight_decay=self.cmd_args.weight_decay
        self.weight=self.cmd_args.weight
        self.nesterov=self.cmd_args.nesterov
        self.lr_decay_milestones=self.cmd_args.lr_decay_milestones
        self.optimizer=self.cmd_args.optimizer
        self.batch_size=self.cmd_args.batch_size
        self.gen_num=self.cmd_args.gen_num
        self.decay_freq=self.cmd_args.decay_freq
        self.save_freq=self.cmd_args.save_freq
        self.print_freq=self.cmd_args.print_freq
        self.save_cp=self.cmd_args.save_cp
        self.cc_feat=self.cmd_args.ccfeat
        self.max_size=self.cmd_args.max_size
        self.num_channel_initial=self.cmd_args.num_channel_initial
        self.group_num=self.cmd_args.group_num
        self.nb_class=self.cmd_args.nb_class
        self.class_index=self.cmd_args.class_index
        self.print_tb=self.cmd_args.print_tb
        self.fold=self.cmd_args.fold
        self.scale=self.cmd_args.scale
        self.components=self.cmd_args.components
        self.task=self.cmd_args.task
        self.val_percent=self.cmd_args.val_percent
        self.out_threshold=self.cmd_args.out_threshold
        self.load=self.cmd_args.load
        self.save_imgs=self.cmd_args.save_imgs
        self.phase=self.cmd_args.phase
        self.data_source=self.cmd_args.data_source
        self.lr=self.cmd_args.lr
        self.net=self.cmd_args.net
        self.grid_size=self.cmd_args.grid_size
        self.span_range=self.cmd_args.span_range

        if self.phase == 'test' or self.phase=='gen' or self.phase=='valid' or self.phase=='visulize'or self.phase=='feat':
            self.load = True
            # self.save_imgs=True
        self.span_range_height = self.span_range_width = self.span_range
        self.grid_height = self.grid_width = self.grid_size
        self.image_height = self.image_width = self.image_size

        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        self.dynamic_config(self.cmd_args)


    def dynamic_config(self, args):
        # dynamica config

        self.model_id = f"wo_segconstraint_myo_tps_{args.net}_{args.data_source}_{args.weight}"
        self.checkpoint_dir = '../outputs/%s/checkpoint' % (self.model_id)
        self.sample_dir = '../outputs/%s/sample' % (self.model_id)
        self.test_dir = '../outputs/%s/test' % (self.model_id)
        self.log_dir = '../outputs/%s/log' % (self.model_id)
        self.res_excel = '../outputs/result/%s.xls'% (self.model_id)
        self.output_dir = '../outputs/%s/res' % (self.model_id)
        self.dataset_dir = '../data/gen_%s/data/' % (args.data_source)
        self.gen_dir = '../outputs/%s/gen_res/' % (self.model_id)