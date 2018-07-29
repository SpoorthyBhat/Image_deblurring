import cv2
import os
import numpy as np
import random
from matplotlib import pyplot as plt

def load_images_from_folder(folder, num=0):
    """
    This function loads images from the folder and returns a list of images

    Arguments:
    folder: a string containing the name of the folder to load images from 

    Outputs:
    images: list of all images in the folder
    """

    images = []
    count = 0
    for filename in os.listdir(folder):
        count = count+1
        if num>0 and count>num:
            break
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def blur_images(images):
    """
    This function takes a list of images and returns a list of blurred images.
    
    Arguments:
    images: List of original images
    
    Output:
    blur_images: list of blurred images
    """
    blur_images = []
    for im in images:
        rand = random.randint(1,10)
        blur = cv2.blur(im, (rand,rand),0)
        blur_images.append(blur)
        
    return blur_images

def create_training_dev_set(folder,size_train, size_dev):
    """
    This function loads images from the folder and the training and dev set of required size

    Arguments:
    folder: a string containing the name of the folder to load images from 
    size_train: number of samples needed in the training set
    size_dev: number of samples needed in the dev set
    
    Outputs:
    X_train: list of training blur images
    Y_train: list of training clear images
    X_dev: list of dev blur images
    Y_dev: list of dev clear images
    """
    images = load_images_from_folder(folder, size_train + size_dev)
    blur = blur_images(images)
    
    if size_train+size_dev != np.asarray(images).shape[0]:
        size_train = 0.8 * np.asarray(images).shape[0]
    X_train1 = blur[0:size_train]
    Y_train1 = images[0:size_train]
    X_dev1 = blur[size_train:]
    Y_dev1 = images[size_train:]
    X_train = []
    X_dev = []
    Y_train = []
    Y_dev = []
    for i in range(len(X_train1)):
        for j in range(3):
            for k in range(3):
                X_train.append(X_train1[i][j*32:(j+1)*32,k*32:(k+1)*32,:])
                Y_train.append(Y_train1[i][j*32:(j+1)*32,k*32:(k+1)*32,:])
    for i in range(len(X_dev1)):
        for j in range(3):
            for k in range(3):
                X_dev.append(X_dev1[i][j*32:(j+1)*32,k*32:(k+1)*32,:])
                Y_dev.append(Y_dev1[i][j*32:(j+1)*32,k*32:(k+1)*32,:])

    return X_train, Y_train, X_dev, Y_dev

