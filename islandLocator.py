
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
parser.add_argument('-r', "--rows", metavar='r', help="number of rows", type=int, default=10)
parser.add_argument('-c', "--columns", metavar='c', help="number of columns", type=int, default=10)
args=parser.parse_args()



try:
    image = cv2.imread(args.image[0])
    height,width,channels=image.shape
    image = cv2.resize(image, (1000,1000))


    xScalingFactor=1000/width
    yScalingFactor=1000/height
except Exception as e:
    print(e)
    raise Exception("File not found")





#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)


firstRowHorizontal=True
islandWidth=5
islandLength=10

class SquareNodeNetwork(NodeNetwork):

    #this will return the four points inside a square
    def getSamplePointsFromSquare(self,topLeft,topRight,bottomLeft,bottomRight,row=0,col=0):
        #multiplier for how far the sample points are from the edge of the square
        if firstRowHorizontal and (row%2)==(col%2) or not firstRowHorizontal and (row%2)!=col%2:
            return []

        xAvg=(topLeft[0]+topRight[0]+bottomLeft[0]+bottomRight[0])/4
        yAvg=(topLeft[1]+topRight[1]+bottomLeft[1]+bottomRight[1])/4

        return [[xAvg,yAvg]]
    
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        return False
    def dataAsString(self):
        string="x,y,color,row,column,isHorizontal,width,height,angle(rad)\n"
        for (rowI, row) in enumerate(self.samplePoints):
            for (vertexI, vertex) in enumerate(row):
                for (pointI, point) in enumerate(vertex):
                    isHorizontal=self.isRowHorizontal(rowI)

                    string+=f"{round(point[0]/xScalingFactor)}, {round(point[1]/yScalingFactor)}, {point[2]}, {rowI}, {vertexI}, {isHorizontal},{islandWidth},{islandLength},{self.getAngle(rowI,vertexI)}\n"

        return string

    def isRowHorizontal(self,rowI):
        return firstRowHorizontal and rowI%2==0 or not firstRowHorizontal and rowI%2==1

    def getAngle(self,row,col):
        thisPoint=self.getPointPosition(row,col)
        if col>0:
            otherPoint=self.getPointPosition(row,col-1)
        else:
            otherPoint=self.getPointPosition(row,col+1)

        angle=math.atan((thisPoint[1]-otherPoint[1])/(thisPoint[0]-otherPoint[0]))
        return angle



    def drawSamplePoints(self,im):
        samplePoints=self.samplePoints
        for (rowI, row) in enumerate(samplePoints):
            for (vertexI, vertex) in enumerate(row):
                for (pointI, point) in enumerate(vertex):
                    #Draw Point based on color
                    if(point[2]==1):
                        color=RED
                    elif(point[2]==0):
                        color=GREEN
                    elif(point[2]==-1):
                        color=BLUE

                    
                    isHorizontal=self.isRowHorizontal(rowI)

                    if rowI%10==0 and vertexI%10==1:
                        angle=self.getAngle(rowI,vertexI)

                        #im=cv2.circle(im,(int(point[0]),int(point[1])),2,color,-1)
                        if isHorizontal:
                            point1=(point[0]-islandLength/2,point[1]-islandWidth/2)
                            point2=(point[0]+islandLength/2,point[1]+islandWidth/2)
                        else:
                            point1=(point[0]-islandWidth/2,point[1]-islandLength/2)
                            point2=(point[0]+islandWidth/2,point[1]+islandLength/2)
                        


                        point1angle=math.atan2(point1[1]-point[1],point1[0]-point[0])
                        point1angle+=angle

                        radius=math.sqrt((point1[0]-point[0])**2+(point1[1]-point[1])**2)

                        point1=(int(point[0]+radius*math.cos(point1angle)),int(point[1]+radius*math.sin(point1angle)))
                        point2=(int(point[0]-radius*math.cos(point1angle)),int(point[1]-radius*math.sin(point1angle)))

                        #cv2.putText(im,str(self.getAngle(rowI,vertexI))[0:6],(int(point[0]),int(point[1])),cv2.FONT_HERSHEY_COMPLEX,0.5,BLACK)
                        cv2.circle(im,point1,2,color,-1)
                        #cv2.circle(im,(-point1[0],-point1[1]),2,color,-1)
                        cv2.circle(im,point2,2,color,-1)
                        #cv2.rectangle(im,point1,point2,color,1)
                    im=cv2.circle(im,(int(point[0]),int(point[1])),2,color,-1)


                    
    
    
    

n=SquareNodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),args.rows, args.columns,image,pointSampleWidth=21)




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
        if(flags==16 or flags==17):
            n.toggleNearestSamplePoint(x,y)
        else:
            n.selectNearestFixedPoint(x,y)
            n.dragging=True
    elif event==cv2.EVENT_MOUSEMOVE:
        n.updateDragging(x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        n.stopDragging()
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass
    lastMouse=(x,y)
    show()

show();
cv2.setMouseCallback('window', mouse_event)

while True:
    key=cv2.waitKey(0)
    if(key==ord("\r")):
        break;
    elif(key==ord("=")):
        islandWidth+=1
        n.setSamplePoints()
    elif(key==ord("-")):
        islandWidth-=1
        n.setSamplePoints()
    elif key==ord("+"):
        islandLength+=1
    elif key==ord("_"):
        islandLength-=1
    elif(key==ord("r")):
        n.addRow()
    elif(key==ord("e")):
        n.removeRow()

    elif(key==ord("c")):
        n.addCol()
    elif(key==ord("x")):
        n.removeCol()
    
    show()


csvName=args.image[0].split(".")[0]+".csv"
with open(csvName, 'w') as file:
    file.write(n.dataAsString())

outputImage=np.zeros((1000,1000,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

cv2.destroyAllWindows()
