import SimpleITK as sitk
import os
import numpy as np
import json
import task_setting

from CommonTools import sitk_tool
from CommonTools.CT_preprocess import get_CT_non_air_bbox


def get_dataset():
    image_dir = r'H:\RawData\MICCAI_Aorta2024\training\images'
    gt_dir = r'H:\RawData\MICCAI_Aorta2024\training\masks_20240918_fixed'

    save_dir = f'{task_setting.path_nnunet_raw_data}/{task_setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    os.makedirs(save_dir_image, exist_ok=True)
    os.makedirs(save_dir_label, exist_ok=True)

    target_spacing = (3.0, None, None)  # (Z, Y, X)
    target_size = (None, 160, 192)

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
        image_nii = sitk_tool.to_orientation(image_nii)
        gt_nii = sitk_tool.to_orientation(gt_nii)

        # Check geometry
        same_geometry = sitk_tool.compare_geometry_multiple([image_nii, gt_nii])
        if not same_geometry:
            raise Exception(f"{case_id} has not same geometry !")

        count += 1
        print(f"==> {count}: {case_id}")

        # Merge GT
        gt = sitk.GetArrayFromImage(gt_nii)
        new_gt = np.zeros_like(gt)
        new_gt[gt == 1] = 1
        new_gt[np.logical_and(gt >= 2, gt <= 7)] = 2
        new_gt[gt == 8] = 3
        new_gt[gt == 9] = 4
        new_gt[np.logical_and(gt >= 10, gt <= 16)] = 5
        new_gt[gt == 17] = 6
        new_gt[np.logical_and(gt >= 18, gt <= 21)] = 7
        new_gt[np.logical_and(gt >= 22, gt <= 23)] = 8

        gt_nii = sitk.GetImageFromArray(new_gt)
        gt_nii = sitk_tool.copy_nii_info(image_nii, gt_nii)

        # Remove air
        non_air_bbox = get_CT_non_air_bbox(image_nii)
        by, ey, bx, ex = non_air_bbox
        image_nii = image_nii[bx:ex+1, by:ey+1, :]
        gt_nii = gt_nii[bx:ex+1, by:ey+1, :]

        # Resampling
        ori_size = gt_nii.GetSize()[::-1]
        ori_spacing = gt_nii.GetSpacing()[::-1]

        new_size = [-1, target_size[1], target_size[2]]
        new_spacing = [ori_size[k] * ori_spacing[k] / new_size[k] for k in range(3)]
        new_spacing[0] = target_spacing[0]
        new_size = [int(ori_size[k] * ori_spacing[k] / new_spacing[k]) for k in range(3)]

        image_nii = sitk_tool.resample(image_nii, new_spacing=new_spacing[::-1], new_size=new_size[::-1], interp=sitk.sitkLinear)
        gt_nii = sitk_tool.resample_to_template(gt_nii, template_nii=image_nii, dtype=sitk.sitkUInt8, constant_value=0)

        # Pseudo spacing
        image_nii.SetSpacing((1.234, 1.234, 1.234))
        gt_nii.SetSpacing((1.234, 1.234, 1.234))

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
            "1": 1,
        },
        "numTraining": numTraining,
        "file_ending": ".nii.gz",
        # "regions_class_order": [1, 2],
    }

    for i in range(1, 9):
        dataset_info['labels'][str(i)] = i

    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=4, sort_keys=False)


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
