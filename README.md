# Image_deblurring

How disappointing is it to have a photo capturing a memorable moment only to find out later that it's blur. Image deblurringcan also be used be used as a preprocessing step for other applications. This project allows you to deblur images.

## Usage

Pre-trained weights and the model used are stored in the repository. You can directly load them and run the deblurring an shown in Demo.ipynb

If you want to train the model from scratch, the trainng script is in deblur.py

## Model

A CNN model with 3 convolution layers has been used. The training set comprises 4000 blur images of size 96x96 and the target set consists of the corresponding clear images. The actual deblurring is learnt on smaller patches of size 32x32.

During prediction, clear patches are predicted from patches of 32x32 at a time.

## Samples

![alt text](https://raw.githubusercontent.com/SpoorthyBhat/Image_deblurring/blur_lenna.png)
