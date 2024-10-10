# -*- encoding: utf-8 -*-
import time

import SimpleITK as sitk
import numpy as np
import cv2
import skimage.morphology
import skimage.measure

from CommonTools import sitk_tool
from CommonTools.bbox import get_bbox


def get_CT_non_air_bbox(image_nii,
                  threshold=-200,
                  process_spacing=(40., 8., 8.), # Z, Y, X
                  erode_iter=1,
                  dilate_iter=1,
                  min_voxel=16
                  ):
    # Resample for fast processing
    resampled_image_nii = sitk_tool.resample(image_nii, new_spacing=process_spacing[::-1])

    # Binarize the image
    b_image = sitk.GetArrayFromImage(resampled_image_nii)
    b_image = np.uint8(b_image >= threshold)

    # Axial view boundary
    final_by = 99999999
    final_ey = -1
    final_bx = 99999999
    final_ex = -1

    kernel = np.ones((3, 3), np.uint8)
    for slice_i in range(b_image.shape[0]):

        cur_slice = b_image[slice_i, :, :]

        cur_slice = cv2.erode(cur_slice, kernel, iterations=erode_iter)
        cur_slice = cv2.dilate(cur_slice, kernel, iterations=dilate_iter)
        cur_slice = skimage.measure.label(cur_slice)
        cur_slice = skimage.morphology.remove_small_objects(cur_slice, min_size=min_voxel, connectivity=1)

        bbox = get_bbox(cur_slice)
        if bbox is not None:
            by, ey, bx, ex = bbox
            final_by = min(final_by, by)
            final_ey = max(final_ey, ey)
            final_bx = min(final_bx, bx)
            final_ex = max(final_ex, ex)

    # reflect to ori image
    ori_spacing = image_nii.GetSpacing()
    ori_size = image_nii.GetSize()
    new_spacing = resampled_image_nii.GetSpacing()

    final_by = max(0, int((final_by - 1) * new_spacing[1] / ori_spacing[1]))
    final_ey = min(ori_size[1]-1, int((final_ey + 1) * new_spacing[1] / ori_spacing[1]))
    final_bx = max(0, int((final_bx - 1) * new_spacing[0] / ori_spacing[0]))
    final_ex = min(ori_size[0]-1, int((final_ex + 1) * new_spacing[0] / ori_spacing[0]))

    # return
    return [final_by, final_ey, final_bx, final_ex]