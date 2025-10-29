# DpDNet: An Dual-Prompt-Driven Network for Universal PET-CT Segmentation

This project extends **nnUNetv2** with **STU-Net** to support **multi-cancer segmentation**.  
Cancer-type information is embedded into the **file naming convention**, and the preprocessing pipeline reads this information automatically so the model can distinguish cancer categories during both training and inference.

---

## 1. Dataset Preparation

Each training case must contain **two input image channels** and **one label file**.  
The cancer type must be encoded in the filename prefix.

### Supported Cancer Types
| Type Prefix | Cancer Category |
|------------|----------------|
| `brea`     | Breast cancer |
| `lymp`     | Lymphoma |
| `mela`     | Melanoma |
| `lung`     | Lung cancer |

---

### 1.1 Image File Naming Format

#### Examples
brea_B000000665_20140331_resampled_image_0000.nii.gz
brea_B000000665_20140331_resampled_image_0001.nii.gz

lymp_B000000665_20140331_resampled_image_0000.nii.gz
lymp_B000000665_20140331_resampled_image_0001.nii.gz

mela_B000000665_20140331_resampled_image_0000.nii.gz
mela_B000000665_20140331_resampled_image_0001.nii.gz

lung_B000000665_20140331_resampled_image_0000.nii.gz
lung_B000000665_20140331_resampled_image_0001.nii.gz


---

### 1.2 Label File Naming Format

#### Examples
brea_B000000665_20140331_resampled_image.nii.gz
lymp_B000000665_20140331_resampled_image.nii.gz
mela_B000000665_20140331_resampled_image.nii.gz
lung_B000000665_20140331_resampled_image.nii.gz

---

## 2. Cancer-Type Extraction Logic

We modify the default nnUNet preprocessing pipeline so that the system recognizes cancer category from the filename prefix.

File modified:
nnunetv2/preprocessing/preprocessors/default_preprocessor.py


This allows each patch to **carry its cancer-type metadata**, enabling the model to dynamically adapt during both training and inference.

---

## 3. Modify Label Mapping in STUNetTrainer

Please edit the cancer-type index mapping based on the number of cancer types in your dataset:

File:
STUNetTrainer.py

python
Copy code

Update:
```python
self.map = {'lymp': 0, 'mela': 1, 'lung': 2, 'brea': 3}
If you add or remove cancer categories, adjust the dictionary accordingly.

4. Training & Preprocessing
All preprocessing and training steps follow the standard nnUNetv2 + STU-Net workflow.

Step 1: Preprocess
bash
Copy code
python nnUNet/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py -d <DATASET_ID> -c 3d_fullres
Step 2: Training
bash
Copy code
python nnUNet/nnunetv2/run/run_training.py Dataset<DATASET_ID>_seg 3d_fullres 0 -tr STUNetTrainer_small_prompt
5. Inference
Inference follows the standard nnUNetv2 inference command.
The model will automatically detect the cancer type based on filename prefix during test-time.

6. Notes
Ensure filename prefixes (brea, lymp, mela, lung) are correct.

Ensure label naming matches the image prefix.

Ensure self.map in STUNetTrainer is consistent with dataset classes.

7. Citation
If you use this work, please consider citing 
@inproceedings{liang2025dpdnet,
  title={DpDNet: An Dual-Prompt-Driven Network for Universal PET-CT Segmentation},
  author={Liang, Xinglong and Huang, Jiaju and Han, Luyi and Zhang, Tianyu and Wang, Xin and Gao, Yuan and Lu, Chunyao and Cai, Lishan and Tan, Tao and Mann, Ritse},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={163--172},
  year={2025},
  organization={Springer}
}
