import torch
from torch.utils.tensorboard import SummaryWriter


class BaseExperiment():
    def write_dict_to_tb(self,dict,step):
        for k in dict.keys():
            self.eval_writer.add_scalar(f"{k}",dict[k],step)

    def __init__(self,args):
        self.args=args
        self.eval_writer = SummaryWriter(log_dir=f"{args.log_dir}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
