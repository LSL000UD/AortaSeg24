import SimpleITK as sitk
import os
import numpy as np
import json
import math

import task_setting

from Utils import utils_simpleitk
from Utils.bbox import get_bbox, extend_bbox


def get_dataset():
    image_dir = r'H:\RawData\MICCAI_Aorta2024\training\images'
    gt_dir = r'H:\RawData\MICCAI_Aorta2024\training\masks_20240918_fixed'

    save_dir = f'{task_setting.path_nnunet_raw_data}/{task_setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    os.makedirs(save_dir_image, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    list_train_case_ids = []
    for file in os.listdir(gt_dir):
        if file.find('_label.mha') == -1:
            continue
        case_id = file.split('_label.mha')[0]

        list_train_case_ids.append(case_id)
    print(f"==> Total {len(list_train_case_ids)} train cases")

    count = 0
    for case_id in list_train_case_ids:

        image_nii = sitk.ReadImage(f"{image_dir}/{case_id}_CTA.mha")
        gt_nii = sitk.ReadImage(f"{gt_dir}/{case_id}_label.mha")

        # To LPS
        image_nii = utils_simpleitk.to_orientation(image_nii)
        gt_nii = utils_simpleitk.to_orientation(gt_nii)

        # Check geometry
        same_geometry = utils_simpleitk.compare_geometry_multiple([image_nii, gt_nii])
        if not same_geometry:
            raise Exception(f"{case_id} has not same geometry !")

        count += 1
        print(f"==> {count}: {case_id}")

        # Get ROI
        gt = sitk.GetArrayFromImage(gt_nii)
        bbox = get_bbox(gt)
        bbox = extend_bbox(bbox, list_extend=(40, 40, 20, 20, 20, 20), max_shape=gt.shape)
        bz, ez, by, ey, bx, ex = bbox

        image_nii = image_nii[bx:ex+1, by:ey+1, bz:ez+1]
        gt_nii = gt_nii[bx:ex+1, by:ey+1, bz:ez+1]

        # Saving
        sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
        sitk.WriteImage(gt_nii, f"{save_dir_label}/{case_id}.nii.gz")

        print(f"        ----> Done ")


def get_dataset_info():
    save_dir = f"{task_setting.path_nnunet_raw_data}/{task_setting.task_name}"
    save_dir_label = f"{task_setting.path_nnunet_raw_data}/{task_setting.task_name}/labelsTr"

    # Train case ids
    numTraining = 0
    for file in os.listdir(save_dir_label):
        if file.find('.nii.gz') > -1:
            numTraining += 1
    print(f"==> Total {numTraining} train cases")

    # Dataset json
    dataset_info = {
        "channel_names": {  # must belong to ['CT', 'noNorm', 'zscore', 'rescale_to_0_1', 'rgb_to_0_1']
            "0": "CT",
        },
        "labels": {
            "background": 0,
        },
        "numTraining": numTraining,
        "file_ending": ".nii.gz",
        # "regions_class_order": [k for k in range(0, 20)],
    }

    for i in range(1, 24):
        dataset_info['labels'][str(i)] = i


    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=4, sort_keys=False)


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
