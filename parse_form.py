import sys
from PIL import Image
import cv2

FORM_PATH = '/home/vimal/Downloads/lead_creation_form.jpg'

def crop_on_border(img_path):
    img = Image.open(img_path)
    nonwhite_positions = [(x,y) for x in range(img.size[0]) for y in range(img.size[1]) if img.getdata()[x+y*img.size[0]] != 255]
    rect = (min([x for x,y in nonwhite_positions]), min([y for x,y in nonwhite_positions]), max([x for x,y in nonwhite_positions]), max([y for x,y in nonwhite_positions]))
    return img.crop(rect) #.save('out.jpg')



def create_binary_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # df = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,100)
    mask_inv = cv2.bitwise_not(mask)
    #denoised = cv2.fastNlMeansDenoising(mask_inv, None, 7, 21, 30)
    return cv2.imwrite('binary_image.jpg', mask)
    


create_binary_image(FORM_PATH)
crop_on_border('binary_image.jpg').save('out.png')
