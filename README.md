# Solution for MICCAI AortaSeg24 challenge
aorta branch Segmentation ， nn-UNet  

## Overview
This solution is based on [nnUnet](https://github.com/MIC-DKFZ/batchgenerators) and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
![image](https://github.com/LSL000UD/AortaSeg24/blob/main/overview.png)

Following figure illustrates the comprehensive framework of our methods, which is mainly based on nnUnet2. The input CT is first processed by a coarse model to roughly
locate the region of interest (ROI) of the aorta. Then, a fine model performs inference to obtain precise segmentation results. To enhance the robustness of
the prediction and reduce the model’s inference time, a sliding-window method taken coarse prediction as input is employed for the fine stage.



## Requirements
- torch==2.0.0+cu117
- Python 3.10
- At least 32 GB GPU memory

- MedPy==0.4.0
- nibabel==3.2.1
- numpy==1.24.1
- opencv-python==4.4.0.46
- pandas==2.0.3
- pydicom==2.1.2
- scikit-image==0.21.0
- scikit-learn==0.24.1
- scipy==1.10.1
- seaborn==0.13.0
- SimpleITK==2.2.1
- sklearn==0.0
- threadpoolctl==3.1.0
- tifffile==2023.7.10
- tqdm==4.53.0
- typing-extensions==4.3.0
- connected-components-3d==3.10.5
- openpyxl==3.1.3

## Code structure

Training codes are directly modified on nn-UNet (Apache-2.0 license),  it may not be well organized.

- acvl_utils: nnUnet-related codes,  https://github.com/MIC-DKFZ/acvl_utils
- batchgenerators: nnUnet-related codes,  https://github.com/MIC-DKFZ/batchgenerators
- dynamic_network_architectures: nnUnet-related codes,  https://github.com/MIC-DKFZ/dynamic-network-architectures
- nnunet: old version of nnUnet,  https://github.com/MIC-DKFZ/nnUNet
- nnunetv2: new version of nnUnet,  https://github.com/MIC-DKFZ/nnUNet
- Training: run nnUnet training
- Testing: run nnUnet testing
- Utils: some usage of nnUnet, SimpleITK, scikit-image.

## Training

- Download [Competition Data](https://aortaseg24.grand-challenge.org/)
	
	All nnUnet tasks share similar training procedures
	- cd Training/Tasks/Task_XXX/
	- python step_0_nnUNet_prepare_raw_data.py
	- python step_1_nnUNet_planning_preprocessing.py
	- python step_3_nnUNet_change_plan.py
	- python step_2_nnUNet_run_training.py
  - 
## Testing

After training, you can run Testing/test.py to test your own cases. 
You can also download our trained models for testing, however, please refer to [https://aortaseg24.grand-challenge.org/)] for the license of this models since our models are trained based on this data.

A Simple usage is: (more details please refer to test.py)

    # Init predictor
    predictor = get_predictor(MODEL_DIR, device)

    # Read input
    image_nii = sitk.ReadImage(INPUT_FILE)

    # Predicting
    pred_nii = predictor.predict_from_nii(image_nii)

## Acknowledgement
-Thank [nnUnet](https://github.com/MIC-DKFZ/batchgenerators), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
and  AortaSeg24 Organizers
	-[https://aortaseg24.grand-challenge.org/)]

