import numpy as np
import SimpleITK as sitk
import os

from Utils.utils_simpleitk import copy_nii_info


def main():
    ori_gt_dir = r"H:\RawData\MICCAI_Aorta2024\training\masks"
    gt_dir = r"H:\RawData\MICCAI_Aorta2024\training\masks_20240918"
    save_dir = r"H:\RawData\MICCAI_Aorta2024\training\masks_20240918_fixed"

    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for file in os.listdir(gt_dir):
        if file.find(".mha") == -1:
            continue

        count += 1
        case_id = file.split("_label.mha")[0]
        print(f"==> {count}: {case_id}" )

        if case_id not in ["subject040"]:
            continue

        ori_gt_nii = sitk.ReadImage(f"{ori_gt_dir}/{file}")
        gt_nii = sitk.ReadImage(f"{gt_dir}/{file}")

        # Error cases
        if case_id == "subject040":
            bx, by, bz = ori_gt_nii.TransformPhysicalPointToIndex(gt_nii.GetOrigin())
            gt = sitk.GetArrayFromImage(gt_nii)
            real_gt = np.zeros(ori_gt_nii.GetSize()[::-1], np.uint8)
            real_gt[bz:bz+gt.shape[0], by:by+gt.shape[1], bx:bx+gt.shape[2]] = gt
            real_gt_nii = sitk.GetImageFromArray(real_gt)
            real_gt_nii = copy_nii_info(ori_gt_nii, real_gt_nii)
            gt_nii = real_gt_nii
        else:
            continue

        sitk.WriteImage(gt_nii, f"{save_dir}/{case_id}_label.mha")


if __name__ == '__main__':
    main()

