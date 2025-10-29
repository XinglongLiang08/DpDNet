import sys
sys.path.append(r'/home/x.liang/MyProject/nnUNet')
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import torch

# from torchsummary import summary




if __name__ == '__main__':

    predictor = nnUNetPredictor(tile_step_size=0.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda', 0),verbose=False,verbose_preprocessing=False,allow_tqdm=True)

    predictor.initialize_from_trained_model_folder('/projects/whole_body_PET_CT_segmentation/nnUNetFrame/DATASET/nnUNet_trained_models/Dataset221_AutoPETII_2023/nnUNetTrainer__nnUNetPlans__3d_fullres',use_folds=(0,),checkpoint_name='checkpoint_final.pth',)
    segmentation_model = predictor.network

    # input_tensor = torch.randn(1, 1, 224, 224)
    
    #[batch_size, channel, depth, height, width]
    input_tensor = torch.randn(1, 2, 128, 128, 128)
    output = segmentation_model(input_tensor)

    print(output)