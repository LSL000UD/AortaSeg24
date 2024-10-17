import json

import task_setting
import os

from nnunetv2.experiment_planning.plan_and_preprocess_api import plan_experiments, preprocess, extract_fingerprints


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

    # Fingerprint extraction
    print("==> Fingerprint extraction...")
    extract_fingerprints(
        dataset_ids=[int(task_setting.task_name.split('_')[1])],
        check_dataset_integrity=True
    )

    # Planning
    print("==> Planning...")
    plan_experiments(
        dataset_ids=[int(task_setting.task_name.split('_')[1])],
        gpu_memory_target_in_gb=12,
    )

    # Pre-processing
    print("==> Processing...")
    preprocess(
        dataset_ids=[int(task_setting.task_name.split('_')[1])],
        configurations=['3d_fullres'],
        num_processes=[4],
    )

    # Changing plans
    print("==> Changing plans ...")
    plan = json.load(open(f"{task_setting.plans_file}", 'r'))

    plan['configurations']['2d'] = None
    plan['configurations']['3d_lowres'] = None
    plan['configurations']['3d_cascade_fullres'] = None

    plan['configurations']['3d_fullres']['batch_size'] = task_setting.batch_size
    plan['configurations']['3d_fullres']['patch_size'] = task_setting.patch_size
    plan['configurations']['3d_fullres']['UNet_base_num_features'] = task_setting.UNet_base_num_features
    plan['configurations']['3d_fullres']['num_pool_per_axis'] = task_setting.num_pool_per_axis
    plan['configurations']['3d_fullres']['pool_op_kernel_sizes'] = task_setting.pool_op_kernel_sizes
    plan['configurations']['3d_fullres']['conv_kernel_sizes'] = task_setting.conv_kernel_sizes
    plan['configurations']['3d_fullres']['batch_dice'] = task_setting.batch_dice

    print(plan)
    json.dump(plan, open(f"{task_setting.plans_file}", 'w'), indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
