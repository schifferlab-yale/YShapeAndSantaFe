import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('height', metavar='image', type=str, nargs='+',help='Path of image')
parser.add_argument('phase', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

#Get the image and make sure it exists
heightIm=[];
try:
    print(args.height[0])
    heightIm = cv2.imread(args.height[0])
    heightIm = cv2.resize(heightIm, (1000,1000))
except:
    raise Exception("File not found")
try:
    phaseIm = cv2.imread(args.phase[0])
    phaseIm = cv2.resize(phaseIm, (1000,1000))
except:
    raise Exception("File not found")


edgeTrim=100

def getAverageColor(im):
    avgColor=0;
    width, height, channels = im.shape
    for i in range(edgeTrim, width-edgeTrim):
        for j in range(edgeTrim, height-edgeTrim):
            avgColor+=(int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3
    avgColor/=(width-2*edgeTrim)*(height-2*edgeTrim);
    return avgColor
avgColor=getAverageColor(phaseIm)
print(avgColor)

def getStdColor(im,avg):
    std=0;
    width, height, channels = im.shape
    for i in range(edgeTrim, width-edgeTrim):
        for j in range(edgeTrim, height-edgeTrim):
            std+=abs((int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3-avgColor)
    std/=(width-2*edgeTrim)*(height-2*edgeTrim);
    return std
stdColor=getStdColor(phaseIm,avgColor)
print(stdColor)

c=1
bias=0
for (rowI, row) in enumerate(heightIm):
    for (pixelI, pixel) in enumerate(row):
        if((int(pixel[0])+int(pixel[1])+int(pixel[2]))/3<5):
            phaseIm[rowI][pixelI]=[127,127,127]
        else:
            if phaseIm[rowI][pixelI][0]+bias>avgColor+stdColor*c:
                phaseIm[rowI][pixelI]=[255,255,255]
            elif phaseIm[rowI][pixelI][0]+bias<avgColor-stdColor*c:
                phaseIm[rowI][pixelI]=[0,0,0]
            else:
                continue
                phaseIm[rowI][pixelI]=[127,127,127]




cv2.imshow("window",heightIm)
cv2.imshow("window2",phaseIm)
cv2.waitKey(0)
cv2.imwrite("out.jpg", np.float32(phaseIm));
cv2.destroyAllWindows()
