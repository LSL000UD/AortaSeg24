import numpy as np


def binary_dice(pred, gt, smooth = 1e-8):

    assert pred.dtype == np.bool_ and gt.dtype == np.bool_, \
        'Input and output should be np.bool_'

    sum_inter = np.sum(np.logical_and(pred, gt))
    sum_gt = np.sum(gt)
    sum_pred = np.sum(pred)

    dice = (2 * sum_inter + smooth) / (sum_gt + sum_pred + smooth)

    return dice

