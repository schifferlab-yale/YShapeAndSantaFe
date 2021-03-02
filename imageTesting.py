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

#CONSTANTS
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(0,0,255)
GRAY=(127,127,127)
def getAverageColor(im):
    avgColor=0;
    width, height, channels = im.shape
    for i in range(width):
        for j in range(height):
            avgColor+=(int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3
    avgColor/=width*height;
    return avgColor
avgColor=getAverageColor(im)

def getStdColor(im,avg):
    std=0;
    width, height, channels = im.shape
    for i in range(width):
        for j in range(height):
            std+=abs((int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3-avgColor)
    std/=width*height;
    return std
stdColor=getStdColor(im,avgColor)

print(avgColor)
print(stdColor)

def getCol(im,x,y):
    return (int(im[x][y][0])+int(im[x][y][1])+int(im[x][y][2]))/3

c=0.8;
for i in range(imWidth):
    for j in range(imHeight):
        col=getCol(im,i,j)
        if(col<avgColor-stdColor*c):
            im[i][j]=BLACK
        elif(col>avgColor+stdColor*c):
            im[i][j]=WHITE
        else:
            im[i][j]=GRAY




imWidth, imHeight, imChannels = im.shape
cv2.imshow("window",im)
cv2.waitKey(0)
cv2.imwrite("out.jpg", np.float32(im));
cv2.destroyAllWindows()
