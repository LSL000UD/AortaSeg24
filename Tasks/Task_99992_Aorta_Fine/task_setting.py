import numpy as np
import os
import sys


# Task Name
task_name = __file__.replace('\\', '/').split('/')[-2]

RUN_LOCAL = os.path.abspath(__file__).find('H:') > -1
if RUN_LOCAL:
    path_root = "XXX"
else:
    path_root = 'XXX'

# Path
path_nnunet = f"{path_root}/nnUNet"
path_nnunet_raw_data = f"{path_nnunet}/nnUNet_raw_data"
path_nnunet_preprocessing_output_dir = f"{path_nnunet}/nnUNet_preprocessed"
path_nnunet_model_dir = f"{path_nnunet}/Models"

dataset_directory = f"{path_nnunet_preprocessing_output_dir}/{task_name}"
plans_file = f"{path_nnunet_preprocessing_output_dir}/{task_name}/nnUNetPlans.json"

# Plan
batch_size = 4
UNet_base_num_features = 32
patch_size = [160, 160, 160]
num_pool_per_axis = [5, 5, 5]
pool_op_kernel_sizes = [[1, 1, 1]] + [[2, 2, 2]] * 5
conv_kernel_sizes = [[3, 3, 3]] * 6
batch_dice = True


# clip_range ={
#         "0": {
#             "percentile_00_5": -160.,
#             "percentile_99_5": 840.,
#         }
#     }

clip_range = None


# Training
list_fold = ['all']
list_GPU_id = [1]
pretrained_weights = None
default_num_threads = 16
default_num_threads_DA = 48
max_num_epochs = 500

# ---------------------------- Modify from original nnunet --------------------------- #
# Set environ
os.environ['nnUNet_raw'] = path_nnunet_raw_data
os.environ['nnUNet_preprocessed'] = path_nnunet_preprocessing_output_dir
os.environ['nnUNet_results'] = path_nnunet_model_dir
os.environ['nnUNet_def_n_proc'] = str(default_num_threads)
os.environ['default_num_threads_DA'] = str(default_num_threads_DA)
# os.environ['CUDA_VISIBLE_DEVICES'] = str(list_GPU_id[0]) + ',' + str(list_GPU_id[1])
os.environ['CUDA_VISIBLE_DEVICES'] = str(list_GPU_id[0])
os.environ['nnUNet_max_num_epochs'] = str(max_num_epochs)
os.environ['nnUNet_sample_foreground_p'] = str(0.5)

# Add src to path
src_path = os.path.abspath(os.path.join(__file__, '../../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

