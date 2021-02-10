import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

#Get the image and make sure it exists
im=[];
try:
    im = cv2.imread(args.image[0])
    im = cv2.resize(im, (1000,1000))
except:
    raise Exception("File not found")

imCopy=im.copy()
imWidth, imHeight, imChannels = im.shape

def mouse_click():
    referencePoints=[]


cv2.imshow("window",im)
cv2.setMouseCallback('window', mouse_click)
cv2.waitKey(0)


# close all the opened windows.
cv2.destroyAllWindows()
