# -*- encoding: utf-8 -*-
import math


def sliding_window_3D(input_size, patch_size, stride):
    """
    :param input_size:  (600, 512, 512)
    :param patch_size: (192, 192, 192)
    :param stride: (92, 92, 92)
    :return:
    """
    list_coords = []
    for d_i in range(3):
        input_l = input_size[d_i]
        output_l = patch_size[d_i]
        stride_l = stride[d_i]

        if input_l <= output_l:
            coords = [0]
        else:
            num_step = math.ceil((input_l - output_l) / stride_l) + 1
            new_stride = (input_l - output_l) / (num_step - 1)
            coords = [int(round(i * new_stride)) for i in range(num_step)]

        list_coords.append(coords)

    output = []
    for z in list_coords[0]:
        for y in list_coords[1]:
            for x in list_coords[2]:
                output_z = patch_size[0]
                output_y = patch_size[1]
                output_x = patch_size[2]
                output.append([z, min(input_size[0]-1, z + output_z - 1),
                               y, min(input_size[1]-1, y + output_y - 1),
                               x, min(input_size[2]-1, x + output_x - 1)])

    return output



