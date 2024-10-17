import abc
import os
import warnings
from warnings import warn

import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import Compose


class CustomTransform(Compose):
    def __init__(self, transforms):
        super(CustomTransform, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")

        for t_i in range(len(self.transforms)):
            t = self.transforms[t_i]

            # Use Nearest augmentation
            if t.__class__.__name__ == 'SpatialTransform':
                self.transforms[t_i].order_data = 0
                self.transforms[t_i].angle_y = (-0, 0)
                self.transforms[t_i].angle_z = (-0, 0)

    def __call__(self, **data_dict):
        for t in self.transforms:
            # data_dict = t(**data_dict)

            list_aug_for_all_channel = ['SpatialTransform', 'MirrorTransform', 'NumpyToTensor']
            if t.__class__.__name__ in list_aug_for_all_channel:
                warnings.warn(f'==> {t.__class__.__name__} will used for all channel !!!!!!!!!!!')
                data_dict = t(**data_dict)
            else:
                tmp = data_dict['data'][:, 1:].copy()
                data_dict = t(**data_dict)
                data_dict['data'][:, 1:] = tmp[:, :]

        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"


class CustomTransformVal(CustomTransform):
    def __init__(self, transforms):
        super(CustomTransformVal, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")
