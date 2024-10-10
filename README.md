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
- Python 3.9
- At least 32 GB GPU memory


## Training

Training codes are directly modified on nn-UNet, so it may not be well organized.

1. Stage 1
   	
	- Download [Competition Data](https://aortaseg24.grand-challenge.org/)
	- Speicfy all path in task_setting.py
	
	- Follow nnUnet workflow to train the model and get prediction results of stage1 
		- cd /Tasks/Task_99991_Aorta_Coarse3mm_160x192/
		- python step_0_nnUNet_prepare_raw_data.py
		- python step_1_nnUNet_planning_preprocessing.py
		- python step_3_nnUNet_change_plan.py
		- python step_2_nnUNet_run_training.py

2. Stage 2 
 This stage using both CT and centerlines as input (for testing, using stage1's predctions to extract centerline)
	- Follow nnUnet workflow to train the model and get prediction results of stage1 
		- cd /Tasks/Task_99992_Aorta_Fine/
		- python step_0_nnUNet_prepare_raw_data.py
		- python step_1_nnUNet_planning_preprocessing.py
		- python step_3_nnUNet_change_plan.py
		- python step_2_nnUNet_run_training.py

 		 - cd /Tasks/Task_99993_Aorta_FineType2/
		- python step_0_nnUNet_prepare_raw_data.py
		- python step_1_nnUNet_planning_preprocessing.py
		- python step_3_nnUNet_change_plan.py
		- python step_2_nnUNet_run_training.py
  - 
## Testing

After training, you can use this notebook for inference https://drive.google.com/file/d/1CkqrrM85v1l6dEuseVa2MoBhpPyNjs9A/view?usp=drive_link

## Acknowledgement
-Thank [nnUnet](https://github.com/MIC-DKFZ/batchgenerators), [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
and  AortaSeg24 Organizers
	-[https://aortaseg24.grand-challenge.org/)]

