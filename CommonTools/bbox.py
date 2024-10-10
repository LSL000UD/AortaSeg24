# -*- encoding: utf-8 -*-
import numpy as np


def get_bbox(mask):
    """
    >> Faster than directly use np.where
    """
    dims = len(mask.shape)

    bbox = [-1, -1] * dims
    for axis_i in np.argsort(mask.shape):
        list_axis = []
        for i in range(dims):
            if i != axis_i:
                list_axis.append(i)

        exist_l = np.where(np.any(mask, axis=tuple(list_axis)) > 0)[0]

        if len(exist_l) == 0:
            return None

        b_l = int(np.min(exist_l))
        e_l = int(np.max(exist_l))

        bbox[2 * axis_i] = int(b_l)
        bbox[2 * axis_i + 1] = int(e_l)

        list_slicer = []
        for i in list_axis:
            list_slicer.append(slice(0, mask.shape[i], None))
        list_slicer.insert(axis_i, slice(b_l, e_l + 1, None))
        mask = mask[tuple(list_slicer)]

    return bbox


def extend_bbox(bbox, list_extend, max_shape):
    dimensions = len(max_shape)

    extended_bbox = []
    for i in range(dimensions):
        extended_bbox.append(max(0, bbox[2 * i] - list_extend[2 * i]))
        extended_bbox.append(min(max_shape[i] - 1, bbox[2 * i + 1] + list_extend[2 * i + 1]))

    return extended_bbox


def extend_bbox_to_size(bbox, target_size, max_shape):
    assert len(bbox) == 2 * len(target_size) == 2* len(max_shape)

    output = []

    num_dim = len(bbox) // 2
    for axis_i in range(num_dim):
        bl = bbox[2*axis_i]
        el = bbox[2*axis_i + 1]
        cur_len = el - bl + 1

        assert max_shape[axis_i] >= target_size[axis_i]

        extend = max(0, target_size[axis_i] - cur_len)
        extend_l = extend // 2
        extend_r = extend - extend_l

        bl = bl - extend_l
        if bl < 0:
            bl = 0
            el = bl + target_size[axis_i] - 1
        else:
            el = el + extend_r
            if el > max_shape[axis_i]-1:
                el = max_shape[axis_i]-1
                bl = el - target_size[axis_i] + 1

        output += [bl, el]

    return output


def extend_bbox_physical(bbox, list_extend_physical, max_shape, spacing, approximate_method=np.ceil):
    """
    :param list_extend: physical length in mm
    :param spacing: mm
    """

    dimensions = len(max_shape)

    list_extend = []
    for i in range(dimensions):
        list_extend.append(int(approximate_method(list_extend_physical[2 * i] / spacing[i])))
        list_extend.append(int(approximate_method(list_extend_physical[2 * i + 1] / spacing[i])))

    extended_bbox = extend_bbox(bbox, list_extend, max_shape)
    return extended_bbox


def merge_bbox_maximum(list_bbox):
    merged_bbox = []
    dims = len(list_bbox[0]) // 2
    num_bbox = len(list_bbox)

    for i in range(dims):
        bl = None
        el = None

        for j in range(num_bbox):
            cur_bbox = list_bbox[j]
            if cur_bbox is None:
                continue

            if bl is None:
                bl = cur_bbox[2 * i]
                el = cur_bbox[2 * i + 1]
            else:
                bl = min(bl, cur_bbox[2 * i])
                el = max(el, cur_bbox[2 * i + 1])

        if bl is None:
            return None

        merged_bbox += [bl, el]

    return merged_bbox


def get_bbox_from_skimage_prop(prop):
    bbox = prop.bbox
    bbox = [bbox[0], bbox[3] - 1, bbox[1], bbox[4] - 1, bbox[2], bbox[5] - 1]

    return bbox


if __name__ == '__main__':
    output = extend_bbox_to_size([10, 20, 10, 20, 20, 30], [40, 40, 40], max_shape=[30, 40, 60])
    print(output)
