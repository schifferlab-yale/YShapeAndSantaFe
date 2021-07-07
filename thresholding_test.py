import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse

from numpy.lib.function_base import average
from nodeNetwork import *

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()


try:
    image = cv2.imread(args.image[0])
    #image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")

bw_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(bw_image,(3,3),1)

avg,std=cv2.meanStdDev(bw_image)
avg=avg[0][0]
std=std[0][0]
print(avg,std)

thresh_image=blurred.copy()
_,thresh_image=cv2.threshold(thresh_image,avg-2*std,255,cv2.THRESH_TOZERO)
_,thresh_image=cv2.threshold(thresh_image, avg+2*std,255,cv2.THRESH_TRUNC)


cv2.imshow("window",thresh_image)
cv2.waitKey(0)