import time
import os
import pickle
import warnings

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import json

import torch
import torch.nn as nn

from nnunet.network_architecture.generic_UNet import Generic_UNet
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from CommonTools import patch_generation
from CommonTools.CT_preprocess import get_CT_non_air_bbox
from CommonTools.IO import find_files_in_dir
from CommonTools import sitk_tool
from CommonTools.bbox import get_bbox, extend_bbox_physical


def print_param(params):
    for key in params:
        print(f"        ----> {key:40s}: {params[key]}")


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

        print(f"==> NNUNetV1Predictor's params are: ")
        print_param(self.infer_params)

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

    def sliding_window_inference(self, image):
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

        # --------------------------------- Sliding window -------------------------------- #
        # Get sliding-window
        input_size = image.shape[1:]
        list_bboxes = patch_generation.sliding_window_3D(
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

                # Flip TTA
                if tta:
                    p_flip_z = (0, 1) if 2 in tta_flip_axis else (0,)
                    p_flip_y = (0, 1) if 3 in tta_flip_axis else (0,)
                    p_flip_x = (0, 1) if 4 in tta_flip_axis else (0,)
                else:
                    p_flip_z = (0,)
                    p_flip_y = (0,)
                    p_flip_x = (0,)

                for flip_z in p_flip_z:
                    for flip_y in p_flip_y:
                        for flip_x in p_flip_x:
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

        # Return prob
        output = output / (count + 1e-8)
        bz, ez, by, ey, bx, ex = pad_bbox
        output = output[:, bz:ez + 1, by:ey + 1, bx:ex + 1]

        output = np.array(output.cpu().data)
        return output.astype(np.float32)


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

        print(f'==> Init plan from {self.plan_file}')
        print(f'==> Init dataset from {self.dataset_file}')

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

                print(f'==> Init model from {self.list_model_pth[i]} to device {self.device}')


class NNUNetCTPredictor:
    def __init__(self,
                 base_predictor,
                 params
                 ):
        self.base_predictor = base_predictor

        self.infer_params = {
            'spacing': self.base_predictor.infer_params['spacing'],
            'resampling_tolerance': 0.01,
            'resampling_mode': sitk.sitkLinear,
            'resampling_dtype': sitk.sitkInt16,
            'resampling_constance_value': -2048,
            'UseNearestNeighborExtrapolator': False,
            'resampling_use_dummy_3D': True,
            'remove_air_CT': False,
            'save_dtype': np.uint8,
            'to_LPS': False,

            'merge_order_for_sigmoid_predictions':None,
            'th_for_sigmoid': 0.5

        }
        for key in params:
            if key not in self.infer_params.keys():
                raise Exception(f"Un-know param {key}")
        self.infer_params.update(params)

        if not isinstance(self.infer_params['resampling_mode'], list):
            self.infer_params['resampling_mode'] = [self.infer_params['resampling_mode']]
        if not isinstance(self.infer_params['resampling_dtype'], list):
            self.infer_params['resampling_dtype'] = [self.infer_params['resampling_dtype']]
        if not isinstance(self.infer_params['resampling_constance_value'], list):
            self.infer_params['resampling_constance_value'] = [self.infer_params['resampling_constance_value']]

        print(f"==> NNUNetCTPredictor's params are: ")
        print_param(self.infer_params)

    def resampling(self, list_image_nii):
        resampling_tolerance = self.infer_params['resampling_tolerance']
        resampling_use_dummy_3D = self.infer_params['resampling_use_dummy_3D']
        resampling_mode = self.infer_params['resampling_mode']
        resampling_dtype = self.infer_params['resampling_dtype']
        resampling_constance_value = self.infer_params['resampling_constance_value']
        UseNearestNeighborExtrapolator = self.infer_params['UseNearestNeighborExtrapolator']
        new_spacing = self.infer_params['spacing'][::-1]  # To (X,Y,Z)

        ori_spacing = list_image_nii[0].GetSpacing()
        do_resampling = np.any(np.abs(np.array(ori_spacing) - np.array(new_spacing)) > resampling_tolerance)
        if do_resampling:
            if resampling_use_dummy_3D:
                for image_i in range(len(list_image_nii)):
                    target_new_spacing = new_spacing
                    target_new_origin = None
                    target_new_size = None
                    target_new_direction = None

                    if image_i >= 1:
                        target_new_spacing = list_image_nii[0].GetSpacing()
                        target_new_origin = list_image_nii[0].GetOrigin()
                        target_new_size = list_image_nii[0].GetSize()
                        target_new_direction = list_image_nii[0].GetDirection()

                    list_image_nii[image_i] = sitk_tool.dummy_3D_resample(
                        list_image_nii[image_i],
                        new_spacing=target_new_spacing,
                        new_origin=target_new_origin,
                        new_size=target_new_size,
                        new_direction=target_new_direction,

                        interp_xy=resampling_mode[image_i],
                        interp_z=sitk.sitkNearestNeighbor,

                        dtype=resampling_dtype[image_i],
                        constant_value=resampling_constance_value[image_i],
                        UseNearestNeighborExtrapolator=UseNearestNeighborExtrapolator
                    )

            else:
                for image_i in range(len(list_image_nii)):
                    target_new_spacing = new_spacing
                    target_new_origin = None
                    target_new_size = None
                    target_new_direction = None

                    if image_i >= 1:
                        target_new_spacing = list_image_nii[0].GetSpacing()
                        target_new_origin = list_image_nii[0].GetOrigin()
                        target_new_size = list_image_nii[0].GetSize()
                        target_new_direction = list_image_nii[0].GetDirection()

                    list_image_nii[image_i] = sitk_tool.resample(
                        list_image_nii[image_i],
                        new_spacing=target_new_spacing,
                        new_origin=target_new_origin,
                        new_size=target_new_size,
                        new_direction=target_new_direction,

                        interp=resampling_mode[image_i],
                        dtype=resampling_dtype[image_i],
                        constant_value=resampling_constance_value[image_i],
                        UseNearestNeighborExtrapolator=UseNearestNeighborExtrapolator
                    )
        else:
            print(f'==> No necessary to do resampling ori {ori_spacing}, new: {new_spacing}')

        return list_image_nii

    def pre_processing(self, image):
        norm_v = self.base_predictor.infer_params['norm_v']

        image = np.float32(image)
        pre_precess_channels = range(image.shape[0])
        for chan_i in pre_precess_channels:
            if self.base_predictor.infer_params['normalization_schemes'][chan_i] != 'NoNormalization':
                p_005 = norm_v[chan_i]['percentile_00_5']
                p_995 = norm_v[chan_i]['percentile_99_5']
                mean_ = norm_v[chan_i]['mean_v']
                std_ = norm_v[chan_i]['std_v']

                image[chan_i] = np.clip(image[chan_i], a_min=p_005, a_max=p_995)
                image[chan_i] = (image[chan_i] - mean_) / (std_ + 1e-7)

        return image

    @staticmethod
    def resampling_back(pred_prob, current_nii_info, ori_nii_info):
        num_cls = pred_prob.shape[0]

        list_pred_per_cls = []
        for i in range(num_cls):
            pred_nii = sitk.GetImageFromArray(pred_prob[i])
            sitk_tool.set_nii_info(pred_nii, current_nii_info)

            pred_nii = sitk_tool.resample(
                pred_nii,
                new_spacing=ori_nii_info['spacing'],
                new_origin=ori_nii_info['origin'],
                new_size=ori_nii_info['size'],
                new_direction=ori_nii_info['direction'],
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0,
                UseNearestNeighborExtrapolator=False
            )
            list_pred_per_cls.append(sitk.GetArrayFromImage(pred_nii)[np.newaxis])

        final_pred = np.concatenate(list_pred_per_cls, axis=0)
        return final_pred

    def post_processing(self, pred):
        activate_func = self.base_predictor.infer_params['activate_func']
        if activate_func == torch.sigmoid:
            if pred.dtype == np.uint8:
                pred = pred >= 128
            else:
                pred = pred >= self.infer_params['th_for_sigmoid']
        else:
            pred = np.argmax(pred, axis=0)
        return pred

    @staticmethod
    def do_something_before_inference(list_image_nii):
        return list_image_nii

    def predict_from_nii(self, list_image_nii, return_nii=True):
        time_start = time.time()

        to_LPS = self.infer_params['to_LPS']
        remove_air_CT = self.infer_params['remove_air_CT']
        save_dtype = self.infer_params['save_dtype']
        activate_func = self.base_predictor.infer_params['activate_func']
        merge_order_for_sigmoid_predictions = self.infer_params['merge_order_for_sigmoid_predictions']

        # Record original nii info before processing
        ori_nii_info = sitk_tool.get_nii_info(list_image_nii[0])
        print(f"                    ----> Image ori info {ori_nii_info}. ")

        # Do something
        list_image_nii = self.do_something_before_inference(list_image_nii)
        print(f"                    ----> Image ori info after do something {sitk_tool.get_nii_info(list_image_nii[0])}. ")

        # to LPS
        if to_LPS:
            for image_i in range(len(list_image_nii)):
                list_image_nii[image_i] = sitk_tool.to_orientation(list_image_nii[image_i], 'LPS')
            print(f"                    ----> Finish to LPS use {time.time() - time_start} seconds. ")

        # Remove air for CT
        if remove_air_CT:
            non_ari_bbox = get_CT_non_air_bbox(list_image_nii[0])
            by, ey, bx, ex = non_ari_bbox

            for image_i in range(len(list_image_nii)):
                list_image_nii[image_i] = list_image_nii[image_i][bx:ex + 1, by:ey + 1, :]

            print(f"            ----> Finish removing CT air use {time.time() - time_start} seconds. ")

        # Resampling
        list_image_nii = self.resampling(list_image_nii)
        print(f"                    ----> Finish resampling use {time.time() - time_start} seconds. ")

        # Pre_processing
        image = []
        for image_i in range(len(list_image_nii)):
            image.append(sitk.GetArrayFromImage(list_image_nii[image_i])[np.newaxis])
        image = np.concatenate(image, axis=0)
        image = self.pre_processing(image)
        print(f"                    ----> Finish pre-processing use {time.time() - time_start} seconds. ")

        # Sliding-window inference
        pred = self.base_predictor.sliding_window_inference(image)
        print(f"                    ----> Finish sliding_window_inference use {time.time() - time_start} seconds. ")

        # Resampling back
        pred = self.resampling_back(pred, sitk_tool.get_nii_info(list_image_nii[0]), ori_nii_info)
        print(f"                    ----> Finish re-sampling back use {time.time() - time_start} seconds. ")

        # Post-Processing
        pred = self.post_processing(pred).astype(save_dtype)
        print(f"                    ----> Finish post processing use {time.time() - time_start} seconds. ")

        if return_nii:
            if activate_func == torch.sigmoid:
                if merge_order_for_sigmoid_predictions is None:
                    outputs = []
                    for chan_i in range(pred.shape[0]):
                        cur_pred_nii = sitk.GetImageFromArray(pred[chan_i])
                        sitk_tool.set_nii_info(cur_pred_nii, ori_nii_info)
                        outputs.append(cur_pred_nii)
                    return outputs
                else:
                    assert isinstance(merge_order_for_sigmoid_predictions, list) or isinstance(merge_order_for_sigmoid_predictions, tuple)
                    final_pred = np.zeros(pred.shape[1:], np.uint8)
                    for chan_i in merge_order_for_sigmoid_predictions:
                        final_pred[pred[chan_i] > 0] = chan_i + 1

                    final_pred_nii = sitk.GetImageFromArray(final_pred)
                    sitk_tool.set_nii_info(final_pred_nii, ori_nii_info)
                    return final_pred_nii
            else:
                pred_nii = sitk.GetImageFromArray(pred)
                sitk_tool.set_nii_info(pred_nii, ori_nii_info)
                return pred_nii
        else:
            return pred

    def predict_from_nii_dir(self,
                             list_input_dir,
                             output_dir,
                             must_include_all=('.nii.gz',),
                             must_include_one_of=None,
                             must_exclude_all=None,

                             list_files=None,
                             sleep_time_between_cases=None
                             ):
        os.makedirs(output_dir, exist_ok=True)

        # Num modalities
        if isinstance(list_input_dir, str):
            list_input_dir = [list_input_dir]

        print(f"Total {len(list_input_dir)} modalities: ")
        for input_dir in list_input_dir:
            print(f"        ----> {input_dir}")

        if list_files is None:
            list_files = find_files_in_dir(list_input_dir[-1],
                                           must_include_all=must_include_all,
                                           must_include_one_of=must_include_one_of,
                                           must_exclude_all=must_exclude_all
                                           )

        count = 0
        print(f"==> Total {len(list_files)} to be inference")
        for file in list_files:
            time_start = time.time()

            file = file.split('/')[-1]
            case_id = file.split('.nii.gz')[0]

            count += 1
            print(f"==> Predicting {count}: {case_id}")

            # Reading
            list_image_nii = []
            for input_dir in list_input_dir:
                image_nii = sitk.ReadImage(f"{input_dir}/{file}")
                list_image_nii.append(image_nii)
            print(f"            ----> Finish reading use {time.time() - time_start} seconds. ")

            # Predicting
            pred_nii = self.predict_from_nii(list_image_nii, return_nii=True)
            print(f"            ----> Finish predicting use {time.time() - time_start} seconds. ")

            # Saving
            # Saving .img.nii.gz using sitk will cause to bug
            if case_id[-4:] == ".img":
                case_id = case_id.replace('.img', '')
            if isinstance(pred_nii, list):
                for chan_i in range(len(pred_nii)):
                    sitk.WriteImage(pred_nii[chan_i], f"{output_dir}/{case_id}_{chan_i}.nii.gz")
            else:
                sitk.WriteImage(pred_nii, f"{output_dir}/{case_id}.nii.gz")
            print(f"            ----> Finish saving use {time.time() - time_start} seconds. ")

            # Sleep to control temperature
            if sleep_time_between_cases is not None:
                time.sleep(sleep_time_between_cases)
                print(f"            ----> Finish sleeping use {time.time() - time_start} seconds. ")


class C2FPredictor:
    def __init__(self, coarse_predictor, find_predictor, extend_mm=(0., 0., 0.), roi_idx_in_coarse=None):
        self.coarse_predictor = coarse_predictor
        self.find_predictor = find_predictor
        self.extend_mm = extend_mm
        self.roi_idx_in_coarse = roi_idx_in_coarse


    def predict_from_nii(self, list_image_nii):
        # Coarse
        pred_coarse_nii = self.coarse_predictor.predict_from_nii(list_image_nii)

        # Get ROI
        pred_coarse = sitk.GetArrayFromImage(pred_coarse_nii)
        if self.roi_idx_in_coarse is None:
            pred_coarse = pred_coarse > 0
        else:
            pred_coarse = pred_coarse == self.roi_idx_in_coarse
        bbox = get_bbox(pred_coarse)
        bbox = extend_bbox_physical(
            bbox,
            list_extend_physical=self.extend_mm,
            max_shape=pred_coarse.shape,
            spacing=list_image_nii[0].GetSpacing()[::-1],
            approximate_method=np.ceil
        )

        bz, ez, by, ey, bx, ex = bbox

        list_roi_image_nii = [list_image_nii[0][bx:ex+1, by:ey+1, bz:ez+1]] + list_image_nii[1:]

        # Fine
        pred_fine_nii = self.find_predictor.predict_from_nii(list_roi_image_nii)
        pred_fine = sitk.GetArrayFromImage(pred_fine_nii)

        final_pred = np.zeros_like(pred_fine)
        final_pred[bz:ez+1, by:ey+1, bx:ex+1] = pred_fine
        final_pred_nii = sitk.GetImageFromArray(final_pred)
        final_pred_nii = sitk_tool.copy_nii_info(list_image_nii[0], final_pred_nii)

        return final_pred_nii


    def predict_from_nii_dir(self,
                             list_input_dir,
                             output_dir,
                             must_include_all=('.nii.gz',),
                             must_include_one_of=None,
                             must_exclude_all=None,

                             list_files=None,
                             sleep_time_between_cases=None
                             ):
        os.makedirs(output_dir, exist_ok=True)

        # Num modalities
        if isinstance(list_input_dir, str):
            list_input_dir = [list_input_dir]

        print(f"Total {len(list_input_dir)} modalities: ")
        for input_dir in list_input_dir:
            print(f"        ----> {input_dir}")

        if list_files is None:
            list_files = find_files_in_dir(list_input_dir[-1],
                                           must_include_all=must_include_all,
                                           must_include_one_of=must_include_one_of,
                                           must_exclude_all=must_exclude_all
                                           )

        count = 0
        print(f"==> Total {len(list_files)} to be inference")
        for file in list_files:
            time_start = time.time()

            file = file.split('/')[-1]
            case_id = file.split('.nii.gz')[0]

            count += 1
            print(f"==> Predicting {count}: {case_id}")

            # Reading
            list_image_nii = []
            for input_dir in list_input_dir:
                image_nii = sitk.ReadImage(f"{input_dir}/{file}")
                list_image_nii.append(image_nii)
            print(f"            ----> Finish reading use {time.time() - time_start} seconds. ")

            # Predicting
            pred_nii = self.predict_from_nii(list_image_nii)
            print(f"            ----> Finish predicting use {time.time() - time_start} seconds. ")

            # Saving
            if isinstance(pred_nii, list):
                for chan_i in range(len(pred_nii)):
                    sitk.WriteImage(pred_nii[chan_i], f"{output_dir}/{case_id}_{chan_i}.nii.gz")
            else:
                sitk.WriteImage(pred_nii, f"{output_dir}/{case_id}.nii.gz")
            print(f"            ----> Finish saving use {time.time() - time_start} seconds. ")

            # Sleep to control temperature
            if sleep_time_between_cases is not None:
                time.sleep(sleep_time_between_cases)
                print(f"            ----> Finish sleeping use {time.time() - time_start} seconds. ")
