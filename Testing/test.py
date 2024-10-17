from predictor import get_predictor
import SimpleITK as sitk
import numpy as np
import torch


if __name__ == '__main__':
    MODEL_DIR = r"H:\Files\MICCAI24_Aorta\Upload\TrainingCodes\Github\AortaSeg24Models"
    INPUT_FILE = r"H:\Files\MICCAI24_Aorta\Upload\Src_V2\LocalInput\images\ct-angiography/subject009_CTA.mha"  # .nii or any other type that SimpleITK can rad
    OUTPUT_FILE = r"G:/subject009_CTA.nii.gz"

    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    # Init predictor
    predictor = get_predictor(MODEL_DIR, device)

    # Read input
    image_nii = sitk.ReadImage(INPUT_FILE)

    # Predicting
    pred_nii = predictor.predict_from_nii(image_nii)

    # Saving
    sitk.WriteImage(pred_nii, f"{OUTPUT_FILE}")
