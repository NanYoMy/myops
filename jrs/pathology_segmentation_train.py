
from nnunet.paths import default_plans_identifier
from nnunet.run.run_training import main as train
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("network")
parser.add_argument("network_trainer")
parser.add_argument("task" ,help="can be task name or task id")
parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
parser.add_argument("--epochs", type=int, default=10, help='-1 default,')
parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                    action="store_true")
parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                    action="store_true")
parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                    default=default_plans_identifier, required=False)
parser.add_argument("--use_compressed_data", default=False, action="store_true",
                    help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                         "is much more CPU and RAM intensive and should only be used if you know what you are "
                         "doing", required=False)
parser.add_argument("--deterministic",
                    help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                         "this is not necessary. Deterministic training will make you overfit to some random seed. "
                         "Don't use that.",
                    required=False, default=False, action="store_true")
parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                      "export npz files of "
                                                                                      "predicted segmentations "
                                                                                      "in the validation as well. "
                                                                                      "This is needed to run the "
                                                                                      "ensembling step so unless "
                                                                                      "you are developing nnUNet "
                                                                                      "you should enable this")
parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                    help="not used here, just for fun")
parser.add_argument("--valbest", required=False, default=False, action="store_true",
                    help="hands off. This is not intended to be used")
parser.add_argument("--fp32", required=False, default=False, action="store_true",
                    help="disable mixed precision training and run old school fp32")
parser.add_argument("--val_folder", required=False, default="validation_raw",
                    help="name of the validation folder. No need to use this for most people")
parser.add_argument("--disable_saving", required=False, action='store_true',
                    help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                         "will be removed at the end of the training). Useful for development when you are "
                         "only interested in the results and want to save some disk space")
parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                    help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                         "closely observing the model performance on specific configurations. You do not need it "
                         "when applying nnU-Net because the postprocessing for this will be determined only once "
                         "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                         "running postprocessing on each fold is computationally cheap, but some users have "
                         "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                         "you should consider setting this flag.")
# parser.add_argument("--interp_order", required=False, default=3, type=int,
#                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
# parser.add_argument("--interp_order_z", required=False, default=0, type=int,
#                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
#                          "Hands off")
# parser.add_argument("--force_separate_z", required=False, default="None", type=str,
#                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                    help='Validation does not overwrite existing segmentations')
parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                    help='do not predict next stage')
parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                    help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                         'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                         'Optional. Beta. Use with caution.')

args = parser.parse_args()

#2d nnUNetTrainerPSNV4 Task633_RJ_MS all
if __name__ == "__main__":
    train(args)

