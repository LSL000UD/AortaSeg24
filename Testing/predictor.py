import time
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm
import skimage.morphology
import skimage.measure

from Utils.nnUNet_base import NNUNetV2Predictor
from Utils.utils_simpleitk import get_nii_info, copy_nii_info, set_nii_info, resample, get_orientation_str, to_orientation
from Utils.bbox import get_bbox, extend_bbox, extend_bbox_to_size, get_CT_non_air_bbox


class CoarseToFinePredictor:
    def __init__(self, dict_path_models, device):
        # init models
        self.predictors = {}
        for task_name in dict_path_models.keys():
            base_predictor = NNUNetV2Predictor(
                list_model_pth=dict_path_models[task_name]["list_model_pth"],
                plan_file=dict_path_models[task_name]["plan_file"],
                dataset_file=dict_path_models[task_name]["dataset_file"],
                device=device,

                params={
                    'tta_flip_axis': (4, 3, 2),
                    'stride': 0.5,
                    'use_gaussian_for_sliding_window': True,
                    'gaussian_sigma_scale': 0.25,
                },
            )
            self.predictors[task_name] = base_predictor

        # Simple test
        with torch.no_grad():
            input_ = torch.zeros((1, 1, 64, 64, 64))
            input_ = input_.to(device)
            output_ = self.predictors["coarse"].list_model[0].forward(input_)

        # Configs
        self.configs = {
            'coarse': {
                'target_spacing': (3.0, None, None),  # (Z, Y, X)
                'target_size': (None, 160, 192)
            },
            'fine': {
                'extend': (40, 40, 20, 20, 20, 20)
            }
        }

    @staticmethod
    def pre_processing(predictor, image):
        norm_v = predictor.infer_params['norm_v']

        image = np.float32(image)
        pre_precess_channels = range(image.shape[0])
        for chan_i in pre_precess_channels:
            if predictor.infer_params['normalization_schemes'][chan_i] != 'NoNormalization':
                p_005 = norm_v[chan_i]['percentile_00_5']
                p_995 = norm_v[chan_i]['percentile_99_5']
                mean_ = norm_v[chan_i]['mean_v']
                std_ = norm_v[chan_i]['std_v']

                image[chan_i] = np.clip(image[chan_i], a_min=p_005, a_max=p_995)
                image[chan_i] = (image[chan_i] - mean_) / (std_ + 1e-7)

        return image

    def predict_coarse(self, image_nii):
        time_start = time.time()

        task_name = "coarse"
        ori_image_info = get_nii_info(image_nii)

        # Remove air
        non_air_bbox = get_CT_non_air_bbox(image_nii)
        by, ey, bx, ex = non_air_bbox
        image_nii = image_nii[bx:ex + 1, by:ey + 1, :]

        # Resampling
        target_spacing = self.configs["coarse"]['target_spacing']
        target_size = self.configs["coarse"]['target_size']

        ori_size = image_nii.GetSize()[::-1]
        ori_spacing = image_nii.GetSpacing()[::-1]

        new_size = [-1, target_size[1], target_size[2]]
        new_spacing = [ori_size[k] * ori_spacing[k] / new_size[k] for k in range(3)]
        new_spacing[0] = target_spacing[0]
        new_size = [int(ori_size[k] * ori_spacing[k] / new_spacing[k]) for k in range(3)]

        image_nii = resample(image_nii, new_spacing=new_spacing[::-1], new_size=new_size[::-1], interp=sitk.sitkLinear)

        # Pre-processing
        image = sitk.GetArrayFromImage(image_nii)
        image = self.pre_processing(self.predictors[task_name], image[np.newaxis])

        # Sliding-window
        pred = self.predictors[task_name].sliding_window_inference(image, list_TTA_axis=[(0, 0, 0)])

        # Resampling back
        pred_nii = sitk.GetImageFromArray(np.uint8(pred))
        pred_nii = copy_nii_info(image_nii, pred_nii)
        pred_nii = resample(
            pred_nii,
            new_spacing=ori_image_info['spacing'],
            new_origin=ori_image_info['origin'],
            new_size=ori_image_info['size'],
            new_direction=ori_image_info['direction'],
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkUInt8,
            constant_value=0
        )

        return pred_nii

    def predict_fine(self, image_nii, pred_coarse_nii):
        time_start = time.time()

        task_name = "fine"
        ori_image_info = get_nii_info(image_nii)

        # Get ROI
        pred_coarse = sitk.GetArrayFromImage(pred_coarse_nii)
        bbox = get_bbox(pred_coarse)
        bbox = extend_bbox(bbox, list_extend=self.configs['fine']['extend'], max_shape=pred_coarse.shape)
        bz, ez, by, ey, bx, ex = bbox
        image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Pre-processing
        image = sitk.GetArrayFromImage(image_nii)
        image = self.pre_processing(self.predictors[task_name], image[np.newaxis])

        # Sliding-window
        pred_coarse = pred_coarse[bz:ez + 1, by:ey + 1, bx:ex + 1]
        pred = self.predictors[task_name].sliding_window_inference(image, pred_coarse=pred_coarse[np.newaxis])

        # Padding back
        final_pred = np.zeros(ori_image_info['size'][::-1], np.uint8)
        final_pred[bz:ez + 1, by:ey + 1, bx:ex + 1] = pred

        final_pred_nii = sitk.GetImageFromArray(np.uint8(final_pred))
        set_nii_info(final_pred_nii, ori_image_info)

        return final_pred_nii

    def predict_cl(self, image_nii, pred_fine_nii):
        time_start = time.time()

        task_name = "cl"
        ori_image_info = get_nii_info(image_nii)

        # Get ROI
        pred_fine = sitk.GetArrayFromImage(pred_fine_nii)
        bbox = get_bbox(pred_fine)
        bbox = extend_bbox(bbox, list_extend=(10, 10, 10, 10, 10, 10), max_shape=pred_fine.shape)
        bbox = extend_bbox_to_size(bbox, target_size=(160, 160, 160 ), max_shape=pred_fine.shape)
        bz, ez, by, ey, bx, ex = bbox

        image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
        pred_fine_nii = pred_fine_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Get skeleton
        pred_fine = sitk.GetArrayFromImage(pred_fine_nii)
        skeleton = np.zeros(pred_fine.shape, np.uint8)
        list_cl_idx = [2, 4, 6, 11, 13, 15, 16, 20, 21, 22, 23]
        for idx in list_cl_idx:
            cur_pred = pred_fine == idx
            cur_bbox = get_bbox(cur_pred)
            if cur_bbox is None:
                continue
            cur_pred = cur_pred[cur_bbox[0]:cur_bbox[1] + 1, cur_bbox[2]:cur_bbox[3] + 1, cur_bbox[4]:cur_bbox[5] + 1]

            cur_skeleton = skimage.morphology.skeletonize(cur_pred)

            skeleton[cur_bbox[0]:cur_bbox[1] + 1, cur_bbox[2]:cur_bbox[3] + 1, cur_bbox[4]:cur_bbox[5] + 1][
                cur_skeleton > 0] = 1
        skeleton = skimage.morphology.binary_dilation(skeleton)

        # Pre-processing
        cur_image = sitk.GetArrayFromImage(image_nii)
        cur_image = np.concatenate([cur_image[np.newaxis], skeleton[np.newaxis]], axis=0)
        cur_image = cur_image.astype(np.float32)
        cur_image = self.pre_processing(self.predictors[task_name], cur_image)

        # Sliding-window
        pred = self.predictors[task_name].sliding_window_inference(
            cur_image,
            list_TTA_axis=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            pred_coarse=pred_fine[np.newaxis]
        )

        # Merge all
        list_merge_idx = [3, 5, 7, 8, 9, 10, 12, 14, 15, 16, 17]
        merged_pred = pred_fine.copy()
        for idx in list_merge_idx:
            merged_pred[merged_pred == idx] = 0
            merged_pred[pred == idx] = idx

        # Pad back
        final_pred = np.zeros(ori_image_info['size'][::-1], np.uint8)
        final_pred[bz:ez+1, by:ey+1, bx:ex+1] = merged_pred

        final_pred_nii = sitk.GetImageFromArray(np.uint8(final_pred))
        set_nii_info(final_pred_nii, ori_image_info)

        return final_pred_nii

    def predict_from_nii(self, image_nii):
        with torch.no_grad():
            ori_orientation_str = get_orientation_str(image_nii)
            image_nii = to_orientation(image_nii, "LPS")

            if get_orientation_str(image_nii) != "LPS":
                raise Exception(f"!!!!!!!! Un-support image direction")
            ori_spacing = image_nii.GetSpacing()
            for dim_i in range(3):
                if abs(ori_spacing[dim_i] - 1) >= 0.05:
                    raise Exception(f"!!!!!!!! Resampling error ")

            time_start = time.time()

            # Coarse
            pred_coarse_nii = self.predict_coarse(image_nii)
            print(f"    ----> Coarse using {time.time() - time_start} seconds")

            # Fine
            pred_fine_nii = self.predict_fine(image_nii, pred_coarse_nii)
            print(f"    ----> Fine using {time.time() - time_start} seconds")

            # Boundary refine
            pred_bd_nii = self.predict_cl(image_nii, pred_fine_nii)
            print(f"    ----> CL using {time.time() - time_start} seconds")

            pred_bd_nii = to_orientation(pred_bd_nii, ori_orientation_str)
            return pred_bd_nii


def get_predictor(MODEL_DIR, device):
    dict_path_models = {
        "coarse": {
            "list_model_pth": [f"{MODEL_DIR}/fold_{k}/coarse.pth" for k in range(3)],
            "plan_file": f'{MODEL_DIR}/coarse_plans.json',
            "dataset_file": f'{MODEL_DIR}/coarse_dataset.json',
        },

        "fine": {
            "list_model_pth": [f"{MODEL_DIR}/fold_{k}/fine.pth" for k in range(4)],
            "plan_file": f'{MODEL_DIR}/fine_plans.json',
            "dataset_file": f'{MODEL_DIR}/fine_dataset.json',

        },
        "cl": {
            "list_model_pth": [f"{MODEL_DIR}/fold_{k}/cl.pth" for k in range(4)],
            "plan_file": f'{MODEL_DIR}/cl_plans.json',
            "dataset_file": f'{MODEL_DIR}/cl_dataset.json',

        },
    }

    predictor = CoarseToFinePredictor(dict_path_models=dict_path_models, device=device)

    return predictor
