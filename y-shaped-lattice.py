import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
from nodeNetwork import *

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
parser.add_argument('-r', "--rows", metavar='r', help="number of rows", type=int, default=10)
parser.add_argument('-c', "--columns", metavar='c', help="number of columns", type=int, default=10)
args=parser.parse_args()

try:
    image = cv2.imread(args.image[0])
    image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")

#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)


class SquareNodeNetwork(NodeNetwork):
    def getSamplePointsFromSquare(self,topLeft,topRight,bottomLeft,bottomRight,row=0):
        #multiplier for how far the sample points are from the edge of the square
        shiftConstant=0.15

        #get center of sides of square
        centerTop=[topLeft[0]+(topRight[0]-topLeft[0])/2, topLeft[1]+(topRight[1]-topLeft[1])/2]
        centerLeft=[topLeft[0]+(bottomLeft[0]-topLeft[0])/2, topLeft[1]+(bottomLeft[1]-topLeft[1])/2]
        centerRight=[topRight[0]+(bottomRight[0]-topRight[0])/2, topRight[1]+(bottomRight[1]-topRight[1])/2]
        centerBottom=[bottomLeft[0]+(bottomRight[0]-bottomLeft[0])/2, bottomLeft[1]+(bottomRight[1]-bottomLeft[1])/2]

        #square width and height
        width=(centerRight[0]-centerLeft[0])
        height=(centerBottom[1]-centerTop[1])

        #sample points are stored as [x,y,color]
        leftSamplePoint=[topLeft[0]+width*shiftConstant, topLeft[1]+height*shiftConstant]
        rightSamplePoint=[topRight[0]-width*shiftConstant, topRight[1]+height*shiftConstant]
        middleSamplePoint=[(centerLeft[0]+centerRight[0])/2,(centerLeft[1]+centerRight[1])/2]
        bottomSamplePoint=[centerBottom[0], centerBottom[1]-height*shiftConstant]

        samplePoints=[leftSamplePoint, rightSamplePoint, middleSamplePoint, bottomSamplePoint]

        if(row%2==1):
            for point in samplePoints:
                point[0]+=width/2



        return samplePoints
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        if(pointI==2):
            sum=0;
            for point in samplePoints[rowI][vertexI]:
                sum+=point[2]
            if(sum==0):
                return False
            else:
                return True
        else:
            return False


n=SquareNodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),args.rows, args.columns,image)
n.pointSampleWidth=1



def show():
    imWidth=1000;
    imHeight=1000;

    outputImage=image.copy()
    n.draw(outputImage)
    cv2.imshow("window",outputImage)

    outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    outputImage[:,:]=(127,127,127)
    n.drawData(outputImage)
    cv2.imshow("output",outputImage)






def mouse_event(event, x, y,flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        n.splitAtClosestPoint(x,y)
    elif event ==cv2.EVENT_LBUTTONDOWN:
        n.selectNearestFixedPoint(x,y)
        n.dragging=True
    elif event==cv2.EVENT_MOUSEMOVE:
        n.updateDragging(x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        n.dragging=False
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass

    show()

show();
cv2.setMouseCallback('window', mouse_event)
cv2.waitKey(0)

with open('output.csv', 'w') as file:
    file.write(n.dataAsString())

outputImage=np.zeros((1000,1000,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

cv2.destroyAllWindows()
