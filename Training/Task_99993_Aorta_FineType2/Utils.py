import os

import numpy as np

from Utils.quick_import import *


def split_left_right(mask):
    mask_l = np.zeros(mask.shape, np.uint8)
    mask_r = np.zeros(mask.shape, np.uint8)

    bbox = get_bbox(mask)

    if bbox is not None:
        bz, ez, by, ey, bx, ex = bbox
        roi_mask = mask[bz:ez + 1, by:ey + 1, bx:ex + 1]
        roi_mask_instance = skimage.measure.label(roi_mask)

        list_area = []
        list_idx = []
        list_center_x = []
        props = skimage.measure.regionprops(roi_mask_instance)
        for prop in props:
            list_area.append(prop.area)
            list_idx.append(prop.label)
            list_center_x.append(prop.centroid[2])

        list_idx = np.array(list_idx)[np.argsort(list_area)[::-1]]
        list_center_x = np.array(list_center_x)[np.argsort(list_area)[::-1]]
        if len(list_idx) >= 2:
            list_idx = list_idx[:2]
            if list_center_x[0] > list_center_x[1]:
                instance_idx_l = list_idx[0]
                instance_idx_r = list_idx[1]
            else:
                instance_idx_l = list_idx[1]
                instance_idx_r = list_idx[0]
        else:
            warnings.warn(f"==> Only pred 1 cc for left-right organ !!!!!!!!!!!!!!!")
            instance_idx_l = list_idx[0]
            instance_idx_r = list_idx[0]

        mask_r[bz:ez + 1, by:ey + 1, bx:ex + 1][roi_mask_instance == instance_idx_r] = 1
        mask_l[bz:ez + 1, by:ey + 1, bx:ex + 1][roi_mask_instance == instance_idx_l] = 1

    return mask_l, mask_r


def try_remove_error_of_GT(gt, case_id):
    ori_gt = gt.copy()

    if case_id == "subject012":
        mask_l, mask_r = split_left_right(gt == 22)
        gt[gt == 22] = 0
        gt[mask_r > 0] = 22
        gt[mask_l > 0] = 21

    # Keep largest CC
    for idx in range(1, 24):
        cur_gt = gt == idx
        bbox = get_bbox(cur_gt)
        if bbox is not None:
            bz, ez, by, ey, bx, ex = bbox
            cur_gt = cur_gt[bz:ez + 1, by:ey + 1, bx:ex + 1]

            cur_gt = keep_largest_cc_binary_3D(cur_gt)

            roi_gt = gt[bz:ez + 1, by:ey + 1, bx:ex + 1]
            roi_gt[roi_gt == idx] = 0
            roi_gt[cur_gt > 0] = idx
        else:
            print(f"        !!!!!!!! {idx} is not exist")

    # Check each label's change
    for idx in range(1, 24):
        cur_gt = gt == idx
        cur_ori_gt = ori_gt == idx
        bbox_gt = get_bbox(cur_gt)
        bbox_ori_gt = get_bbox(ori_gt)
        bbox = merge_bbox_maximum([bbox_gt, bbox_ori_gt])

        if bbox is None:
            print(f"        !!!!!!!!! {idx} not exist")
        else:
            bz, ez, by, ey, bx, ex = bbox
            cur_gt = cur_gt[bz:ez+1, by:ey+1, bx:ex+1]
            cur_ori_gt = cur_ori_gt[bz:ez+1, by:ey+1, bx:ex+1]

            dice = binary_dice(cur_gt, cur_ori_gt)
            if dice != 1:
                print(f"         !!!!!!!! {idx} DICE is : {dice}")

    return gt


def main():
    gt_dir = r"H:\RawData\MICCAI_Aorta2024\training\masks"
    save_dir = r"H:\RawData\MICCAI_Aorta2024\training\masks_fixed"

    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for file in os.listdir(gt_dir):
        if file.find(".mha") == -1:
            continue

        count += 1
        case_id = file.split("_label.mha")[0]
        print(f"==> {count}: {case_id}" )

        # if case_id != "subject012":
        #     continue

        gt_nii = sitk.ReadImage(f"{gt_dir}/{file}")
        gt = sitk.GetArrayFromImage(gt_nii)
        gt = try_remove_error_of_GT(gt, case_id)

        new_gt_nii = sitk.GetImageFromArray(np.uint8(gt))
        new_gt_nii = copy_nii_info(gt_nii, new_gt_nii)

        sitk.WriteImage(new_gt_nii, f"{save_dir}/{case_id}_label.mha")

if __name__ == '__main__':
    main()

