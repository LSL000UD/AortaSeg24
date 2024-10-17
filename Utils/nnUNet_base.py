import time
import pickle
import warnings
import json
import numpy as np
from tqdm import tqdm
import math
from scipy.ndimage.filters import gaussian_filter

import torch
import torch.nn as nn


from dynamic_network_architectures.architectures.unet import PlainConvUNet
from Utils.bbox import get_bbox, extend_bbox, extend_bbox_to_size


def print_param(params):
    for key in params:
        print(f"        ----> {key:40s}: {params[key]}")


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


def sliding_window_3D_split_z(mask, input_size, patch_size, stride, list_extend=(0, 0, 20, 20, 20, 20)):
    """
    :param input_size:  (600, 512, 512)
    :param patch_size: (192, 192, 192)
    :param stride: (92, 92, 92)
    :return:
    """
    output = []

    # Split Z first
    list_bbox_z = sliding_window_3D(input_size, (patch_size[0], input_size[1], input_size[2]), stride)

    # Split Y,Z
    for i in range(len(list_bbox_z)):
        bz, ez, _, _, _, _ = list_bbox_z[i]

        roi_bbox = get_bbox(mask[bz:ez+1])
        if roi_bbox is not None:
            roi_bbox = extend_bbox(roi_bbox, list_extend=list_extend, max_shape=mask.shape)
            roi_bbox = extend_bbox_to_size(roi_bbox, patch_size, mask.shape)

            list_bbox_yx = sliding_window_3D(
                input_size=(roi_bbox[1]-roi_bbox[0]+1, roi_bbox[3]-roi_bbox[2]+1, roi_bbox[5]-roi_bbox[4]+1),
                patch_size=(roi_bbox[1]-roi_bbox[0]+1, patch_size[1], patch_size[2]),
                stride=stride
            )

            for bbox in list_bbox_yx:
                _, _, by, ey, bx, ex = bbox
                output.append([bz, ez, by+roi_bbox[2], ey+roi_bbox[2], bx+roi_bbox[4], ex+ roi_bbox[4]])

    return output


class NNUNetV1Predictor:
    def __init__(self,
                 list_model_pth,
                 plan_file,
                 device,
                 params
                 ):
        self.list_model_pth = list_model_pth
        self.plan_file = plan_file
        self.device = device

        # Init inference params
        self.infer_params = {
            'plan_stage': -1,
            'use_gaussian_for_sliding_window': True,
            'patch_size': None,
            'spacing': None,
            'stride': None,
            'tta_flip_axis': None,
            'model_class': None,
            'activate_func': torch.softmax,
            'use_half_for_prob_tensor': True,
            'num_class': None,
            'norm_v': [],
            'normalization_schemes': [],
            'gaussian_sigma_scale':0.125,
            'sleep_between_each_forward':None
        }
        for key in params:
            if key not in self.infer_params.keys():
                raise Exception(f"Un-know param {key}")
        self.infer_params.update(params)

        self.plan = None
        self.list_model = None
        self.gaussian_map = None

        self.init_plan()
        self.init_model()

        #print(f"==> NNUNetV1Predictor's params are: ")
        #print_param(self.infer_params)

    def init_plan(self):
        self.plan = pickle.load(open(self.plan_file, 'rb'))
        print(f'==> Init plan from {self.plan_file}')

        # Plan stage
        if self.plan['plans_per_stage'] is not None and self.infer_params['plan_stage'] == -1:
            self.infer_params['plan_stage'] = len(self.plan['plans_per_stage']) - 1

        # Num class
        if self.infer_params['num_class'] is None:
            self.infer_params['num_class'] = self.plan['num_classes'] + 1

        # Model class
        if self.infer_params['model_class'] is None:
            self.infer_params['model_class'] = Generic_UNet

        # Patch size
        if self.infer_params['patch_size'] is None:
            self.infer_params['patch_size'] = self.plan['plans_per_stage'][self.infer_params['plan_stage']][
                'patch_size']

        # Stride
        if self.infer_params['stride'] is None:
            self.infer_params['stride'] = [max(1, k // 2) for k in self.infer_params['patch_size']]
        if isinstance(self.infer_params['stride'], float):
            self.infer_params['stride'] = [max(1, int(round(k * self.infer_params['stride']))) for k in
                                           self.infer_params['patch_size']]

        # Spacing
        self.infer_params['spacing'] = self.plan['plans_per_stage'][self.infer_params['plan_stage']]['current_spacing']

        # Normalization params
        for chan_i in range(len(self.plan['dataset_properties']['intensityproperties'])):
            cur_norm_v = {
                'percentile_00_5': self.plan['dataset_properties']['intensityproperties'][chan_i]['percentile_00_5'],
                'percentile_99_5': self.plan['dataset_properties']['intensityproperties'][chan_i]['percentile_99_5'],
                'mean_v': self.plan['dataset_properties']['intensityproperties'][chan_i]['mean'],
                'std_v': self.plan['dataset_properties']['intensityproperties'][chan_i]['sd'],
            }
            self.infer_params['norm_v'].append(cur_norm_v)
            self.infer_params['normalization_schemes'].append(self.plan['configurations']['3d_fullres']['normalization_schemes'][chan_i])

    def init_model(self):
        plan_stage = self.infer_params['plan_stage']

        num_input_channels = self.plan['num_modalities']
        base_num_features = self.plan['base_num_features']
        num_classes = self.plan['num_classes'] + 1
        net_numpool = len(self.plan['plans_per_stage'][plan_stage]['pool_op_kernel_sizes'])
        conv_per_stage = self.plan['conv_per_stage']
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        net_num_pool_op_kernel_sizes = self.plan['plans_per_stage'][plan_stage]['pool_op_kernel_sizes']
        net_conv_kernel_sizes = self.plan['plans_per_stage'][plan_stage]['conv_kernel_sizes']

        with torch.no_grad():
            self.list_model = []
            for i in range(len(self.list_model_pth)):
                model = self.infer_params['model_class'](num_input_channels, base_num_features, num_classes,
                                                         net_numpool,
                                                         conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs,
                                                         dropout_op,
                                                         dropout_op_kwargs,
                                                         net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, None,
                                                         net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False,
                                                         True, True)

                ckpt = torch.load(self.list_model_pth[i], map_location='cpu')['state_dict']
                model.load_state_dict(ckpt)
                model.eval()
                model = model.to(self.device)
                self.list_model.append(model)

                print(f'==> Init model from {self.list_model_pth[i]} to device {self.device}')

    def init_gaussian(self):
        patch_size = self.infer_params['patch_size']
        sigma_scale = self.infer_params['gaussian_sigma_scale']

        with torch.no_grad():
            output = np.zeros(patch_size, np.float32)

            # Get sigma and center
            sigmas = []
            center = []
            for dim_i in range(len(patch_size)):
                sigmas.append(patch_size[dim_i] * sigma_scale)
                center.append(patch_size[dim_i] // 2)

            # Get guassian
            output[tuple(center)] = 1
            output = gaussian_filter(output, sigmas, 0, mode='constant', cval=0)

            # Scale and clip
            output = output / np.max(output) * 1000.
            output[output < 0.1] = 0.1

            output = output[np.newaxis, np.newaxis].astype(np.float32)

            self.gaussian_map = torch.from_numpy(output).to(self.device)

    @staticmethod
    def center_pad_to_size_3D(image, target_size, constant_value=0, return_bbox=True):
        # Image is C,Z,Y,X
        ori_shape = image.shape[1:]

        list_pad_l = []
        list_bbox = []
        for axis_i in range(3):
            pad_l = target_size[axis_i] - ori_shape[axis_i]

            pad_l = max(0, pad_l)
            pad_l_1 = pad_l // 2
            pad_l_2 = pad_l - pad_l_1

            list_pad_l += [pad_l_1, pad_l_2]
            list_bbox += [pad_l_1, pad_l_1 + ori_shape[axis_i] - 1]

        image = np.pad(
            image,
            ((0, 0), (list_pad_l[0], list_pad_l[1]), (list_pad_l[2], list_pad_l[3]), (list_pad_l[4], list_pad_l[5])),
            mode='constant',
            constant_values=constant_value
        )

        if return_bbox:
            return image, list_bbox
        else:
            return image

    def model_forward(self, model, patch_input):
        activate_func = self.infer_params['activate_func']

        pred = model.forward(patch_input)

        if activate_func == torch.softmax:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)

        return pred

    def sliding_window_inference(self, image, argmax=True, th_num_bbox=36, pred_coarse=None, list_TTA_axis=None):
        patch_size = self.infer_params['patch_size']
        num_class = self.infer_params['num_class']
        stride = self.infer_params['stride']
        use_half_for_prob_tensor = self.infer_params['use_half_for_prob_tensor']
        tta_flip_axis = self.infer_params['tta_flip_axis']
        use_gaussian_for_sliding_window = self.infer_params['use_gaussian_for_sliding_window']
        tta = tta_flip_axis is not None

        # Try init gaussian map
        if self.infer_params['use_gaussian_for_sliding_window'] and self.gaussian_map is None:
            self.init_gaussian()

        # Pad it if input_size < target_size
        image, pad_bbox = self.center_pad_to_size_3D(
            image,
            target_size=patch_size,
            constant_value=0,
            return_bbox=True
        )

        ##################################
        if pred_coarse is not None:
            pred_coarse, pad_bbox = self.center_pad_to_size_3D(
                pred_coarse,
                target_size=patch_size,
                constant_value=0,
                return_bbox=True
            )

        # --------------------------------- Sliding window -------------------------------- #
        # Get sliding-window
        input_size = image.shape[1:]

        list_bboxes = sliding_window_3D(
            input_size=input_size,
            patch_size=patch_size,
            stride=stride
        )
        
        ##############################
        if pred_coarse is not None:
            list_bboxes = sliding_window_3D_split_z(
                pred_coarse[0],
                input_size=input_size,
                patch_size=patch_size,
                stride=stride
            )

        output_size = (num_class,) + input_size

        if use_half_for_prob_tensor:
            prob_dtype = torch.half
        else:
            prob_dtype = torch.float32

        output = torch.zeros(output_size, dtype=prob_dtype).to(self.device)
        count = torch.zeros(input_size, dtype=prob_dtype).to(self.device)

        # Model Predicting
        with torch.no_grad():
            for bbox in tqdm(list_bboxes):

                bz, ez, by, ey, bx, ex = bbox[:]
                patch_input_ori = image[:, bz:ez + 1, by:ey + 1, bx:ex + 1].copy()

                if list_TTA_axis is None:
                    list_TTA_axis = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]

                if len(list_bboxes) > 12 and len(list_TTA_axis) >= 3:
                    list_TTA_axis = [(0, 0, 0), (1, 0, 0)]
                    print(f"!!!!!!!! Image too long {len(list_bboxes)}, will use less TTA")

                for TTA_axis in list_TTA_axis:
                    flip_z = TTA_axis[0]
                    flip_y = TTA_axis[1]
                    flip_x = TTA_axis[2]


                    patch_input = torch.from_numpy(patch_input_ori).to(self.device).unsqueeze(0)

                    # Get flip axis
                    flip_axis = []
                    if flip_z == 1:
                        flip_axis.append(2)
                    if flip_y == 1:
                        flip_axis.append(3)
                    if flip_x == 1:
                        flip_axis.append(4)

                    # Flip aug
                    do_flip = (flip_z == 1) or (flip_y == 1) or (flip_x == 1)
                    if do_flip:
                        patch_input = torch.flip(patch_input, dims=flip_axis)

                    for model in self.list_model:
                        pred = self.model_forward(model, patch_input)

                        if self.infer_params['sleep_between_each_forward'] is not None:
                            warnings.warn(f"==> Sleep between each model forward !")
                            time.sleep(self.infer_params['sleep_between_each_forward'])

                        # Flip back
                        if do_flip:
                            pred = torch.flip(pred, dims=flip_axis)

                        if use_gaussian_for_sliding_window:
                            pred = pred * self.gaussian_map
                            output[:, bz:ez + 1, by:ey + 1, bx:ex + 1] += pred[0]
                            count[bz:ez + 1, by:ey + 1, bx:ex + 1] += self.gaussian_map[0, 0]

                        else:
                            output[:, bz:ez + 1, by:ey + 1, bx:ex + 1] += pred[0]
                            count[bz:ez + 1, by:ey + 1, bx:ex + 1] += 1.0

        # Prob
        output = output / (count + 1e-8)
        bz, ez, by, ey, bx, ex = pad_bbox
        output = output[:, bz:ez + 1, by:ey + 1, bx:ex + 1]

        # Return mask
        if argmax:
            output = torch.argmax(output, dim=0)
            output = np.array(output.cpu().data)
            return output.astype(np.uint8)
        else:
            output = 255. * output
            output = np.array(output.cpu().data)
            return output.astype(np.uint8)


class NNUNetV2Predictor(NNUNetV1Predictor):
    def __init__(self,
                 list_model_pth,
                 plan_file,
                 dataset_file,
                 device,

                 params
                 ):
        self.dataset_file = dataset_file
        self.dataset_json = None

        super(NNUNetV2Predictor, self).__init__(list_model_pth, plan_file, device, params)

    def init_plan(self):
        self.plan = json.load(open(self.plan_file, 'r'))
        self.dataset_json = json.load(open(self.dataset_file, 'r'))

        # Plan stage
        if self.infer_params['plan_stage'] is None or self.infer_params['plan_stage'] == -1:
            self.infer_params['plan_stage'] = '3d_fullres'

        # Num class and activate_func
        if self.infer_params['num_class'] is None:
            num_class = len(self.dataset_json['labels'].keys())

            has_region = False
            for key in self.dataset_json['labels'].keys():
                if isinstance(self.dataset_json['labels'][key], list) or isinstance(self.dataset_json['labels'][key],
                                                                                    tuple):
                    has_region = True
                    break
            if has_region:
                num_class -= 1
                self.infer_params['activate_func'] = torch.sigmoid

            self.infer_params['num_class'] = num_class

        # Patch size
        if self.infer_params['patch_size'] is None:
            self.infer_params['patch_size'] = self.plan['configurations'][self.infer_params['plan_stage']]['patch_size']

        # Stride
        if self.infer_params['stride'] is None:
            self.infer_params['stride'] = [max(1, k // 2) for k in self.infer_params['patch_size']]
        if isinstance(self.infer_params['stride'], float):
            self.infer_params['stride'] = [max(1, int(round(k * self.infer_params['stride']))) for k in
                                           self.infer_params['patch_size']]

        # Spacing
        self.infer_params['spacing'] = self.plan['configurations'][self.infer_params['plan_stage']]['spacing']

        # Normalization params
        for chan_i in range(len(self.plan['foreground_intensity_properties_per_channel'])):
            cur_norm_v = {
                'percentile_00_5': self.plan['foreground_intensity_properties_per_channel'][str(chan_i)][
                    'percentile_00_5'],
                'percentile_99_5': self.plan['foreground_intensity_properties_per_channel'][str(chan_i)][
                    'percentile_99_5'],
                'mean_v': self.plan['foreground_intensity_properties_per_channel'][str(chan_i)]['mean'],
                'std_v': self.plan['foreground_intensity_properties_per_channel'][str(chan_i)]['std'],
            }
            self.infer_params['norm_v'].append(cur_norm_v)
            self.infer_params['normalization_schemes'].append(self.plan['configurations']['3d_fullres']['normalization_schemes'][chan_i])
        # Model class
        if self.infer_params['model_class'] is None:
            self.infer_params['model_class'] = PlainConvUNet

        #print(f'==> Init plan from {self.plan_file}')
        #print(f'==> Init dataset from {self.dataset_file}')

    def init_model(self):
        plan_stage = self.infer_params['plan_stage']
        model_class = self.infer_params['model_class']
        num_class = self.infer_params['num_class']

        model_type = self.plan['configurations'][plan_stage]['UNet_class_name']
        conv_or_blocks_per_stage = {
            'n_conv_per_stage': self.plan['configurations'][plan_stage]['n_conv_per_stage_encoder'],
            'n_conv_per_stage_decoder': self.plan['configurations'][plan_stage]['n_conv_per_stage_decoder']
        }
        kwargs = {
            model_type: {
                'conv_bias': True,
                'norm_op': nn.InstanceNorm3d,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None, 'dropout_op_kwargs': None,
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
            }
        }

        num_stages = len(self.plan['configurations'][plan_stage]['conv_kernel_sizes'])
        with torch.no_grad():
            self.list_model = []
            for i in range(len(self.list_model_pth)):
                model = model_class(
                    input_channels=len(self.plan['foreground_intensity_properties_per_channel'].keys()),
                    n_stages=num_stages,
                    features_per_stage=[
                        min(self.plan['configurations'][plan_stage]['UNet_base_num_features'] * 2 ** i,
                            self.plan['configurations'][plan_stage]['unet_max_num_features']) for i in
                        range(num_stages)],
                    conv_op=nn.Conv3d,
                    kernel_sizes=self.plan['configurations']['3d_fullres']['conv_kernel_sizes'],
                    strides=self.plan['configurations']['3d_fullres']['pool_op_kernel_sizes'],
                    num_classes=num_class,
                    deep_supervision=False,
                    **conv_or_blocks_per_stage,
                    **kwargs[model_type]
                )

                ckpt = torch.load(self.list_model_pth[i], map_location='cpu')

                model.load_state_dict(ckpt['network_weights'])
                model.eval()
                model = model.to(self.device)
                self.list_model.append(model)

                #print(f'==> Init model from {self.list_model_pth[i]} to device {self.device}')
