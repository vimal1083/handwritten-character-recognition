from glob import glob
import os
import shutil 
import cv2
from random import randint
from tqdm import tqdm
from functools import reduce

DATASET_LOCATION = '/home/aspire/Downloads/RBL/English/Hnd/Img'


def save_variant(variant, img_path, prefix):
    new_img_file_path = os.path.join(os.path.dirname(img_path), prefix + '_' + os.path.basename(img_path))
    cv2.imwrite(new_img_file_path, variant)

def generate_variant(img_path, background, prefix):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    rows, cols, shape = img.shape
    # Create background image
    # bg_img = img.copy()
    # bg_img[0:rows, 0:cols] = background
    # fg_img = cv2.bitwise_or(img, img, mask=mask)
    # bg_img = cv2.bitwise_or(bg_img, bg_img, mask=mask_inv)
    # variant_wbg = cv2.add(fg_img, bg_img)
    # variant_gray = cv2.cvtColor(variant_wbg, cv2.COLOR_BGR2GRAY)

    # blur the image
    variant_blurred = blur_image(mask)

    # save variant image
    save_variant(variant_blurred, img_path, prefix)

def blur_image(img):
    return img # cv2.GaussianBlur(img, (3, 7), randint(0, 2))

def get_background():
    background = (randint(110, 255), randint(110, 255), randint(110, 255))
    prefix = reduce(lambda x, y: str(x) + '_' +str(y), background)
    return background, prefix

def main():
    for folder in tqdm(glob(DATASET_LOCATION + '/*'), desc='Creating Variants'):
        if os.path.isdir(folder):
            for img_path in glob(folder + '/img*.png'):
                generate_variant(img_path, *get_background())

def resize_all():
    for folder in tqdm(glob(DATASET_LOCATION + '/*'), desc='Resizing Images'):
        if os.path.isdir(folder):
            for img_path in glob(folder + '/img*.png'):
                img_file = cv2.imread(img_path)
                cv2.imwrite(img_path, cv2.resize(img_file, None, fx=0.025, fy=0.025))

def delete_orig_image():
    for folder in tqdm(glob(DATASET_LOCATION + '/*'), desc='Deleting Original Images'):
        if os.path.isdir(folder):
            for img_path in glob(folder + '/img*.png'):                
                os.remove(img_path)

if __name__ == '__main__':
    resize_all()
    for k in tqdm(range(10), desc='Completed iterations'):
        main()
    delete_orig_image()

