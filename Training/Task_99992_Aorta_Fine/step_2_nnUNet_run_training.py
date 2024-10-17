import task_setting

import json
import os
import torch

from nnunetv2.run.run_training import run_training


def main():
    print(f"==> nnunet environ:")
    nnunet_environ_keys = [
        'nnUNet_raw',
        'nnUNet_preprocessed',
        'nnUNet_results',
        'CUDA_VISIBLE_DEVICES',
        'nnUNet_def_n_proc',
        'nnUNet_max_num_epochs',
    ]
    for key in nnunet_environ_keys:
        print(f"        ----> {key:30s}: {os.environ.get(key)}")

    # multithreading in torch doesn't help nnU-Net if run on GPU
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(f'cuda:{task_setting.list_GPU_id[0]}')

    # Do training
    for fold in task_setting.list_fold:
        run_training(
            dataset_name_or_id=task_setting.task_name.split('_')[1],
            fold=fold,
            configuration='3d_fullres',
            pretrained_weights=task_setting.pretrained_weights,
            num_gpus=len(task_setting.list_GPU_id),
            continue_training=True,
            device=device
        )


if __name__ == '__main__':
    main()
