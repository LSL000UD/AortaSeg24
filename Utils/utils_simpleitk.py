# -*- encoding: utf-8 -*-
import math
import numpy as np
import warnings
import SimpleITK as sitk

"""
All in (x, y, z) order, eg. the idx and physical point
"""


def copy_nii_info(source_image, target_image):
    # Check same size
    source_size = source_image.GetSize()
    target_size = target_image.GetSize()

    if (source_size[0] != target_size[0]) or (source_size[1] != target_size[1]) or (source_size[2] != target_size[2]):
        warnings.warn(
            f"==> Source image size {source_size} != target image size {target_size} when copying image info !")

    # Doing the copy
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetDirection(source_image.GetDirection())
    target_image.SetOrigin(source_image.GetOrigin())

    return target_image


def get_nii_info(image_nii):
    image_info = {
        'spacing': image_nii.GetSpacing(),
        'direction': image_nii.GetDirection(),
        'origin': image_nii.GetOrigin(),
        'size': image_nii.GetSize()

    }
    return image_info


def set_nii_info(source_image, image_info, size_checking=True):
    # Check same size
    source_size = source_image.GetSize()
    target_size = image_info['size']

    if size_checking:
        if (source_size[0] != target_size[0]) or (source_size[1] != target_size[1]) or (source_size[2] != target_size[2]):
            warnings.warn(
                f"==> Source image size {source_size} != target image size {target_size} when copying image info !")

    source_image.SetSpacing(image_info['spacing'])
    source_image.SetDirection(image_info['direction'])
    source_image.SetOrigin(image_info['origin'])

    return source_image


def resample(
        image_nii,
        new_spacing,
        new_origin=None,
        new_size=None,
        new_direction=None,
        interp=sitk.sitkNearestNeighbor,
        dtype=sitk.sitkInt16,
        constant_value=-2048,
        UseNearestNeighborExtrapolator=False
):
    ori_spacing = image_nii.GetSpacing()
    ori_origin = image_nii.GetOrigin()
    ori_direction = image_nii.GetDirection()
    ori_size = image_nii.GetSize()

    if new_direction is None:
        new_direction = ori_direction

    if new_size is None:
        new_size = [math.ceil(ori_spacing[i] * ori_size[i] / new_spacing[i]) for i in range(3)]

    if new_origin is None:
        new_origin = ori_origin

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputOrigin(new_origin)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(new_direction)
    resample_filter.SetDefaultPixelValue(constant_value)
    resample_filter.SetOutputPixelType(dtype)

    if UseNearestNeighborExtrapolator:
        resample_filter.SetUseNearestNeighborExtrapolator(True)

    resampled_image_nii = resample_filter.Execute(image_nii)

    return resampled_image_nii


def resample_to_template(
        image_nii,
        template_nii,
        interp=sitk.sitkNearestNeighbor,
        dtype=sitk.sitkInt16,
        constant_value=-2048,
        UseNearestNeighborExtrapolator=False
):
    resampled_image_nii = resample(
        image_nii,
        template_nii.GetSpacing(),
        new_origin=template_nii.GetOrigin(),
        new_size=template_nii.GetSize(),
        new_direction=template_nii.GetDirection(),
        interp=interp,
        dtype=dtype,
        constant_value=constant_value,
        UseNearestNeighborExtrapolator=UseNearestNeighborExtrapolator
    )

    return resampled_image_nii


def to_orientation(image, orientation='LPS'):
    """
    Reorient image to orientation eg. LPS, RAI
    :param image: Input itk image.
    :return: Input image reoriented.
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)
    reoriented = orient_filter.Execute(image)

    return reoriented


def get_orientation_str(image):
    return sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(image.GetDirection())


def compare_geometry(image_1, image_2, threshold=0.001, return_each=False):
    same_size = np.all(np.array(image_1.GetSize()) == np.array(image_2.GetSize()))
    same_origin = np.all(np.abs(np.array(image_1.GetOrigin()) - np.array(image_2.GetOrigin())) < threshold)
    same_spacing = np.all(np.abs(np.array(image_1.GetSpacing()) - np.array(image_2.GetSpacing())) < threshold)
    same_direction = np.all(np.abs(np.array(image_1.GetDirection()) - np.array(image_2.GetDirection())) < threshold)

    if return_each:
        return same_size, same_origin, same_spacing, same_direction
    else:
        return same_size and same_origin and same_spacing and same_direction


def compare_geometry_multiple(list_images, threshold=0.001):
    sum_image = len(list_images)
    assert sum_image >= 2

    same_geometry = True
    for i in range(1, sum_image):
        cur_same_geometry = compare_geometry(list_images[0], list_images[i], threshold=threshold, return_each=False)
        if not cur_same_geometry:
            same_geometry = False
            break

    return same_geometry