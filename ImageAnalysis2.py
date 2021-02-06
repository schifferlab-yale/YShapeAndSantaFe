import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

WHITE=(255,255,255)
BLACK=(0,0,0)

im = cv2.imread(args.image[0])
imCopy=im.copy()
imWidth, imHeight, imChannels = im.shape



def mouse_click(event, x, y,flags, param):
    global circles, dragging, selectedPoint
    if event == cv2.EVENT_LBUTTONDOWN:
        selectedPoint=getClosestReferencePoint(x,y)
        dragging=True;
    if event == cv2.EVENT_LBUTTONUP:
        dragging=False;
    elif event == cv2.EVENT_MOUSEMOVE:
        if(dragging):
            referencePoints[selectedPoint]=[x,y]
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass
    showScreen();


referencePoints=[[10,10],[100,10],[10,100],[100,100],[20,10]]
selectedPoint=-1;
dragging=False;

def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))
def getClosestReferencePoint(x,y):
    minDist=100000;
    selected=0;
    for (i,point) in enumerate(referencePoints):
        distance=dist([x,y],point)
        if(distance<minDist):
            selected=i;
            minDist=distance;
    return selected

def showScreen():
    global circles
    im=imCopy.copy();
    cv2.imshow("window",im)
    for point in referencePoints:
        c=cv2.circle(im,(point[0],point[1]), 10, (0,0,255,0.1), 3)
        cv2.imshow("window",c)


showScreen();
cv2.setMouseCallback('window', mouse_click)

cv2.waitKey(0)
# close all the opened windows.
cv2.destroyAllWindows()
