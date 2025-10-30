# Retina-Blood-Vessel-Segmentation-in-PyTorch

This repository contains the code for semantic segmentation of the retina blood vessel on the DRIVE dataset using the PyTorch framework.

The following models are used:
- [UNET](https://arxiv.org/abs/1505.04597)

Models to be used in future:
- RESUNET
- DEEPLABV3+

# Dataset
The ISIC-2018 dataset is used for this for training the UNET architecture. The dataset contains the 2596 pairs of images and masks. All of these images are of different shapes and contains a variety of skin lesions.

Original Image             |  Mask Image
:-------------------------:|:-------------------------:
![](final_results/img_0.png)  |  ![](final_results/result_0.png)
