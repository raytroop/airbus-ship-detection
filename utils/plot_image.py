import os
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .rle import rle_decode

def display_trnImg(img_sample_nme, dataframe, image_shape=(768, 768)):
    print(dataframe[dataframe.ImageId == img_sample_nme])
    img_sample_path = os.path.join('dataset/train_v2/', img_sample_nme)
    img_sample = io.imread(img_sample_path)
    _, axs = plt.subplots(figsize=(10, 10))
    axs.imshow(img_sample)
    if dataframe[dataframe.ImageId == img_sample_nme].EncodedPixels.notnull().sum() > 0:
        rles = dataframe[dataframe.ImageId == img_sample_nme].EncodedPixels.tolist()
        mask = np.zeros(image_shape, dtype=np.uint8)
        print('Total {:2d} ship in image'.format(len(rles)))
        for rle in rles:
            mask += rle_decode(rle, mask_shape=image_shape, mask_dtype=np.uint8)
        axs.imshow(mask, alpha=0.5, cmap='Purples')
