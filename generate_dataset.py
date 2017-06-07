from glob import glob
import os
import shutil 
import cv2

DATASET_LOCATION = '/home/aspire/Downloads/RBL/English/Hnd/Img'

for folder in glob(DATASET_LOCATION + '/*'):
    if os.path.isdir(folder):
        for img_file in glob(folder + '/*'):
            shutil.copy(img_file, os.path.join( os.path.dirname(img_file), 'x' + os.path.basename(img_file) ))
    