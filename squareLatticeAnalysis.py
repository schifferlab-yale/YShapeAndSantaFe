"""
This is a file meant to read data from a square lattice MFM sample. Make sure that the nodeNetwork file is available for it to use.
"""


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

    #this will return the four points inside a square
    def getSamplePointsFromSquare(self,topLeft,topRight,bottomLeft,bottomRight,row=0,col=0):
        #multiplier for how far the sample points are from the edge of the square
        shiftConstant=0.25

        #get center of sides of square
        centerTop=[topLeft[0]+(topRight[0]-topLeft[0])/2, topLeft[1]+(topRight[1]-topLeft[1])/2]
        centerLeft=[topLeft[0]+(bottomLeft[0]-topLeft[0])/2, topLeft[1]+(bottomLeft[1]-topLeft[1])/2]
        centerRight=[topRight[0]+(bottomRight[0]-topRight[0])/2, topRight[1]+(bottomRight[1]-topRight[1])/2]
        centerBottom=[bottomLeft[0]+(bottomRight[0]-bottomLeft[0])/2, bottomLeft[1]+(bottomRight[1]-bottomLeft[1])/2]

        #square width and height
        width=(centerRight[0]-centerLeft[0])
        height=(centerBottom[1]-centerTop[1])

        #sample points are stored as [x,y,color]
        topSamplePoint=[centerTop[0],centerTop[1]+height*shiftConstant]
        leftSamplePoint=[centerLeft[0]+width*shiftConstant,centerLeft[1]]
        rightSamplePoint=[centerRight[0]-width*shiftConstant,centerRight[1]]
        bottomSamplePoint=[centerBottom[0], centerBottom[1]-height*shiftConstant]

        fourSamplePoints=[topSamplePoint,leftSamplePoint,rightSamplePoint,bottomSamplePoint]
        return fourSamplePoints
    
    #this shows when two sides are both black/white which means the data is being read wrong
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        if pointI==0 or pointI==1:
            return False

        row = samplePoints[rowI]
        vertex=row[vertexI]

        if pointI==2:
            if vertexI < len(row)-1:
                if(vertex[2][2] == row[vertexI+1][1][2]):
                    return True
        #if we are not in the last row, make sure bottom ponit is the opposite color than the one below it
        if pointI==3:
            if rowI < len(samplePoints)-1:
                if(vertex[3][2] == samplePoints[rowI+1][vertexI][0][2]):
                    return True
        return False

    
    def drawData(self,im):
        if not self.dragging:
            samplePoints=self.samplePoints

            height, width, channels = im.shape

            margin=50
            vertexVSpacing=(height-2*margin)/(len(samplePoints)-1)
            vertexHSpacing=(width-2*margin)/(len(samplePoints[0])-1)

            spacing=min(vertexVSpacing, vertexHSpacing)
            vertexVSpacing=spacing
            vertexHSpacing=spacing

            islandSpacing=(vertexVSpacing+vertexHSpacing)/6
            for (rowI, row) in enumerate(samplePoints):
                for (vertexI, vertex) in enumerate(row):
                    vertexX=int(margin+vertexI*vertexHSpacing)
                    vertexY=int(margin+rowI*vertexVSpacing)

                    if(vertex[0][2]+vertex[1][2]+vertex[2][2]+vertex[3][2]!=0):
                        im=cv2.circle(im,(int(vertexX),int(vertexY)),3,RED,-1)
                    elif(vertex[0][2]+vertex[1][2]!=0 or vertex[0][2]+vertex[2][2]!=0):
                        im=cv2.circle(im,(int(vertexX),int(vertexY)),3,BLUE,-1)
                    for (pointI, point) in enumerate(vertex):

                        if(point[2]==1):
                            color=WHITE
                        else:
                            color=BLACK

                        if(pointI==0):
                            x=vertexX
                            y=vertexY-islandSpacing
                        elif(pointI==1):
                            x=vertexX-islandSpacing
                            y=vertexY
                        elif(pointI==2):
                            x=vertexX+islandSpacing
                            y=vertexY
                        elif(pointI==3):
                            x=vertexX
                            y=vertexY+islandSpacing

                        if(rowI==0 and pointI==0):
                            #im=cv2.circle(im,(int(point[0]),int(point[1])),4,color,-1)
                            pass
                        elif(vertexI == len(row)-1 and pointI==2):
                            #im=cv2.circle(im,(int(point[0]),int(point[1])),4,color,-1)
                            pass
                        elif(vertexI ==0 and pointI==1):
                            #im=cv2.circle(im,(int(point[0]),int(point[1])),4,color,-1)
                            pass
                        elif(rowI==len(samplePoints)-1 and pointI==3):
                            #im=cv2.circle(im,(int(point[0]),int(point[1])),4,color,-1)
                            pass

                        elif(pointI==2 or pointI==3):
                            if(pointI==2):
                                otherPoint=row[vertexI+1][1]
                                otherX=x+vertexHSpacing-2*islandSpacing
                                otherY=y
                            else:
                                otherPoint=samplePoints[rowI+1][vertexI][0]
                                otherX=x
                                otherY=y+vertexVSpacing-2*islandSpacing

                            if(otherPoint[2]==1):
                                otherColor=WHITE
                            else:
                                otherColor=BLACK

                            im=cv2.line(im, (int(x),int(y)), (int(otherX),int(otherY)), BLACK, 2)

                            ##midPoint=[(point[0]+otherPoint[0])/2, (point[1]+otherPoint[1])/2]
                            ##im=cv2.line(im, (int(point[0]),int(point[1])), (int(midPoint[0])-1, int(midPoint[1])), color, 3)
                            ##im=cv2.line(im, (int(midPoint[0]), int(midPoint[1])), (int(otherPoint[0])+1,int(otherPoint[1])),  otherColor, 3)"""
                        im=cv2.circle(im,(int(x),int(y)),3,color,-1)


n=SquareNodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),args.rows, args.columns,image)




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





lastMouse=(0,0)
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
        n.setSamplePoints()
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass
    lastMouse=(x,y)
    show()

show();
cv2.setMouseCallback('window', mouse_event)
#TODO add a button to cycle a point (correct errors manually)
while True:
    key=cv2.waitKey(0)
    if(key==ord("\r")):
        break;
    elif(key==ord("r")):
        n.addRow()
    elif(key==ord("e")):
        n.removeRow()

    elif(key==ord("c")):
        n.addCol()
    elif(key==ord("x")):
        n.removeCol()
    
    show()

with open('output.csv', 'w') as file:
    file.write(n.dataAsString())

outputImage=np.zeros((1000,1000,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

cv2.destroyAllWindows()
