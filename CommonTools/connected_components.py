# -*- encoding: utf-8 -*-
import skimage.measure
import skimage.morphology
import numpy as np
import cv2
from CommonTools.bbox import get_bbox, get_bbox_from_skimage_prop
import scipy.ndimage as ndi
import warnings

def get_top_n_cc_3D(mask, top_n, merge_top_n=False):
    # Crop bbox for speeding
    bbox = get_bbox(mask)
    if bbox is None:
        return None

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex = bbox[:]
    patch_mask = mask[roi_bz:roi_ez+1, roi_by:roi_ey+1, roi_bx:roi_ex+1]

    # Top n props
    instance_label = skimage.measure.label(patch_mask)
    props = skimage.measure.regionprops(instance_label)
    list_area = []
    for prop in props:
        list_area.append(prop.area)
    sorted_prop_indexes = list(np.argsort(list_area)[::-1])

    list_mask = []
    list_bbox = []
    list_size = []
    if merge_top_n:
        merge_mask = np.zeros(mask.shape, np.uint8)

    for index_i in range(min(top_n, len(props))):
        prop = props[sorted_prop_indexes[index_i]]

        bbox = get_bbox_from_skimage_prop(prop)
        bz, ez, by, ey, bx, ex  = bbox[:]

        list_bbox.append([bz+roi_bz, ez+roi_bz, by+roi_by, ey+roi_by, bx+roi_bx, ex+roi_bx])
        list_size.append(prop.area)

        cur_mask = np.zeros(mask.shape, np.uint8)
        cur_mask[roi_bz+bz:roi_bz+ez+1, roi_by+by:roi_by+ey+1, roi_bx+bx:roi_bx+ex+1][instance_label[bz:ez+1, by:ey+1, bx:ex+1] == prop.label] = 1
        list_mask.append(cur_mask)

        if merge_top_n:
            merge_mask[roi_bz+bz:roi_bz+ez+1, roi_by+by:roi_by+ey+1, roi_bx+bx:roi_bx+ex+1][instance_label[bz:ez+1, by:ey+1, bx:ex+1] == prop.label] = 1

    if merge_top_n:
        return list_mask, merge_mask, list_bbox, list_size
    else:
        return list_mask, list_bbox, list_size


def keep_cc_connect_to_cc_binary_3D(mask, connect_cc, min_size=0.):
    final_output = np.zeros(mask.shape, np.bool_)

    # Crop bbox for speeding
    mask = mask > 0
    bbox = get_bbox(mask)
    if bbox is None:
        return final_output

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex = bbox[:]
    patch_mask = mask[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1]
    patch_mask = remove_small_objects_binary_3D(patch_mask, min_size=min_size)

    patch_connect_cc = connect_cc[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1]
    patch_connect_cc_dilated = skimage.morphology.binary_dilation(patch_connect_cc)
    instance_label = skimage.measure.label(patch_mask)

    list_overlap_idx =  list(np.unique(instance_label[patch_connect_cc_dilated > 0]))

    for idx in list_overlap_idx:
        if idx != 0:
            final_output[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1][instance_label == idx] = 1

    return final_output


def keep_largest_cc_binary_3D(mask, print_all_area=False):
    final_output = np.zeros(mask.shape, np.bool_)

    # Crop bbox for speeding
    mask = mask > 0
    bbox = get_bbox(mask)
    if bbox is None:
        return final_output

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex  = bbox[:]
    patch_mask = mask[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1]

    instance_label = skimage.measure.label(patch_mask)
    props = skimage.measure.regionprops(instance_label)

    # Sort by area
    list_area = []
    for prop in props:
        list_area.append(prop.area)
    sorted_prop_indexes = list(np.argsort(list_area)[::-1])

    final_output[roi_bz:roi_ez+1, roi_by:roi_ey+1, roi_bx:roi_ex+1] = instance_label == props[sorted_prop_indexes[0]].label

    if print_all_area and len(list_area) > 1:
        print(f"CC area is : {list_area}")
    return final_output


def remove_small_objects_binary_3D(label, min_size=64, connectivity=1):
    bbox = get_bbox(label)
    if bbox is None:
        return label

    bz, ez, by, ey, bx, ex  = bbox

    final_output = np.zeros(label.shape, np.bool_)
    roi_label = label[bz:ez + 1, by:ey + 1, bx:ex + 1] > 0
    roi_label = skimage.measure.label(roi_label)
    roi_label = skimage.morphology.remove_small_objects(roi_label, min_size=min_size, connectivity=connectivity)

    final_output[bz:ez + 1, by:ey + 1, bx:ex + 1] = roi_label > 0
    return final_output