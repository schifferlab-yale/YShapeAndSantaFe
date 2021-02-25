import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint



try:
    image = cv2.imread("squareLattice.jpeg")
    image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")


WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)

def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))
def getIntermediate(point1,point2,percent):
    return [(point2[0]-point1[0])*percent+point1[0], (point2[1]-point1[1])*percent+point1[1]]

def sampleImageColor(im,x,y):
    if(im[int(y)][int(x)][0]>127):
        return 1
    else:
        return -1


class Node:
    def __init__(self,x,y):
        self.x=x;
        self.y=y;
    def xyAsIntTuple(self):
        return (int(self.x),int(self.y))
    def xyAsArray(self):
        return [self.x,self.y]
    def drawLineTo(self,node,im):
        im=cv2.line(im, self.xyAsIntTuple(), node.xyAsIntTuple(), RED, 2)
    def __str__(self):
        return "("+str(self.x)+","+str(self.y)+")"

def xyArrayToIntTuple(arr):
    return (int(arr[0]),int(arr[1]))



class NodeNetwork:
    def __init__(self,topLeft,topRight,bottomLeft,bottomRight,rows, cols):

        self.rows=rows;
        self.cols=cols;

        self.selectedPoint=None;
        self.dragging=False

        #format {row: rowIndex, col:colIndex, node:Node}
        self.fixedPoints=[
        {"row":0,"col":0,"node":topLeft},
        {"row":0,"col":cols-1,"node":topRight},
        {"row":rows-1,"col":0,"node":bottomLeft},
        {"row":rows-1, "col":cols-1, "node":bottomRight}
            ]

        self.topLeft=self.fixedPoints[0]["node"];
        self.topRight=self.fixedPoints[1]["node"];
        self.bottomLeft=self.fixedPoints[2]["node"];
        self.bottomRight=self.fixedPoints[3]["node"];

        #rows and columns which contain fixed points
        self.fixedRows=[0,rows-1]
        self.fixedCols=[0,cols-1]



    def draw(self,im):
        #self.topLeft.drawLineTo(self.topRight, im)
        #self.topRight.drawLineTo(self.bottomRight, im)
        #self.bottomRight.drawLineTo(self.bottomLeft, im)
        #self.bottomLeft.drawLineTo(self.topLeft, im)

        #for row in self.getPoints():
            #for point in row:
                #im=cv2.circle(im,(int(point[0]),int(point[1])), 2,BLACK,-1)

        sortedPoints=self.getSortedFixedPoints()
        for (i,point) in enumerate(self.getSortedFixedPoints()):
            if self.selectedPoint==point:
                im=cv2.circle(im,point["node"].xyAsIntTuple(), 10,RED,-1)
            else:
                im=cv2.circle(im,point["node"].xyAsIntTuple(), 5,RED,-1)
            hasGrid=True;
            topLeft=point
            topLeftCoord=point["node"].xyAsArray()
            if(i==len(sortedPoints)-1):#bottomLeftCorner
                hasGrid=False
            elif(point["row"]==self.rows-1):#bottomRow
                hasGrid=False
            elif(point["col"]==self.cols-1):#last col
                hasGrid=False
            else:#has a valid grid
                topRight=sortedPoints[i+1]
                bottomLeft=sortedPoints[i+len(self.fixedCols)]
                bottomRight=sortedPoints[i+len(self.fixedCols)+1]
                cols=topRight["col"]-topLeft["col"]
                rows=bottomLeft["row"]-topLeft["row"]

                topRightCoord=topRight["node"].xyAsArray()
                bottomLeftCoord=bottomLeft["node"].xyAsArray()
                bottomRightCoord=bottomRight["node"].xyAsArray()

            if(hasGrid):
                im=cv2.line(im,xyArrayToIntTuple(topLeftCoord),xyArrayToIntTuple(topRightCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(topLeftCoord),xyArrayToIntTuple(bottomLeftCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(bottomLeftCoord),xyArrayToIntTuple(bottomRightCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(topRightCoord),xyArrayToIntTuple(bottomRightCoord),BLACK,1)

                for (startX,startY,endX,endY) in zip(np.linspace(topLeftCoord[0],topRightCoord[0],cols+1), np.linspace(topLeftCoord[1],topRightCoord[1],cols+1), np.linspace(bottomLeftCoord[0],bottomRightCoord[0],cols+1), np.linspace(bottomLeftCoord[1],bottomRightCoord[1],cols+1)):
                    im=cv2.line(im,(int(startX),int(startY)),(int(endX),int(endY)),RED,1)
                for (startX,startY,endX,endY) in zip(np.linspace(topLeftCoord[0],bottomLeftCoord[0],rows+1), np.linspace(topLeftCoord[1],bottomLeftCoord[1],rows+1), np.linspace(topRightCoord[0],bottomRightCoord[0],rows+1), np.linspace(topRightCoord[1],bottomRightCoord[1],rows+1)):
                    im=cv2.line(im,(int(startX),int(startY)),(int(endX),int(endY)),RED,1)

        if not self.dragging:
            samplePoints=self.getSamplePoints()
            for (rowI, row) in enumerate(samplePoints):
                for (vertexI, vertex) in enumerate(row):
                    for (pointI, point) in enumerate(vertex):
                        #Draw Point
                        if(point[2]==1):
                            im=cv2.circle(im,(int(point[0]),int(point[1])),2,RED,-1)
                        else:
                            im=cv2.circle(im,(int(point[0]),int(point[1])),2,BLUE,-1)

                        #Check for errors:

                    #if we are not in the last column
                    if vertexI < len(row)-1:
                        if(vertex[2][2] == row[vertexI+1][1][2]):
                            im=cv2.circle(im,(int(vertex[2][0]),int(vertex[2][1])),5,GREEN,2)
                    if rowI < len(samplePoints)-1:
                        if(vertex[3][2] == samplePoints[rowI+1][vertexI][0][2]):
                            im=cv2.circle(im,(int(vertex[3][0]),int(vertex[3][1])),5,GREEN,2)


    def getGrid(self):
        #generate an empty grid representing each point
        grid=[]
        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):
                grid[len(grid)-1].append("")

        sortedPoints=self.getSortedFixedPoints()
        for (i,point) in enumerate(self.getSortedFixedPoints()):
            hasGrid=True;
            topLeft=point
            topLeftCoord=point["node"].xyAsArray()
            if(i==len(sortedPoints)-1):#bottomLeftCorner
                hasGrid=False
            elif(point["row"]==self.rows-1):#bottomRow
                hasGrid=False
            elif(point["col"]==self.cols-1):#last col
                hasGrid=False
            else:#has a valid grid
                topRight=sortedPoints[i+1]
                bottomLeft=sortedPoints[i+len(self.fixedCols)]
                bottomRight=sortedPoints[i+len(self.fixedCols)+1]
                cols=topRight["col"]-topLeft["col"]
                rows=bottomLeft["row"]-topLeft["row"]

                topRightCoord=topRight["node"].xyAsArray()
                bottomLeftCoord=bottomLeft["node"].xyAsArray()
                bottomRightCoord=bottomRight["node"].xyAsArray()

            if(hasGrid):
                for (rowI, rowStartX,rowStartY,rowEndX,rowEndY) in  zip(range(0,rows+1), np.linspace(topLeftCoord[0],bottomLeftCoord[0],rows+1), np.linspace(topLeftCoord[1],bottomLeftCoord[1],rows+1), np.linspace(topRightCoord[0],bottomRightCoord[0],rows+1), np.linspace(topRightCoord[1],bottomRightCoord[1],rows+1)):
                    for(colI, pointX, pointY) in zip(range(0,cols+1), np.linspace(rowStartX,rowEndX,cols+1), np.linspace(rowStartY,rowEndY,cols+1)):
                        grid[point["row"]+rowI][point["col"]+colI]=[pointX,pointY]
        return grid


    #get sample points as [row][vertex][island] which is an array [x,y,color]
    def getSamplePoints(self):
        shiftConstant=0.25
        grid=self.getGrid()
        samplePoints=[]
        for (rowI, row) in enumerate(grid[:-1]):
            samplePoints.append([])
            for (pointI,point) in enumerate(row[:-1]):

                topLeft=point
                topRight=grid[rowI][pointI+1]
                bottomLeft=grid[rowI+1][pointI]
                bottomRight=grid[rowI+1][pointI+1]

                centerTop=[topLeft[0]+(topRight[0]-topLeft[0])/2, topLeft[1]+(topRight[1]-topLeft[1])/2]
                centerLeft=[topLeft[0]+(bottomLeft[0]-topLeft[0])/2, topLeft[1]+(bottomLeft[1]-topLeft[1])/2]
                centerRight=[topRight[0]+(bottomRight[0]-topRight[0])/2, topRight[1]+(bottomRight[1]-topRight[1])/2]
                centerBottom=[bottomLeft[0]+(bottomRight[0]-bottomLeft[0])/2, bottomLeft[1]+(bottomRight[1]-bottomLeft[1])/2]

                width=(centerRight[0]-centerLeft[0])
                height=(centerBottom[1]-centerTop[1])

                #sample points are stored as [x,y,color]
                topSamplePoint=[centerTop[0],centerTop[1]+height*shiftConstant]
                leftSamplePoint=[centerLeft[0]+width*shiftConstant,centerLeft[1]]
                rightSamplePoint=[centerRight[0]-width*shiftConstant,centerRight[1]]
                bottomSamplePoint=[centerBottom[0], centerBottom[1]-height*shiftConstant]

                fourSamplePoints=[topSamplePoint,leftSamplePoint,rightSamplePoint,bottomSamplePoint]
                for samplePoint in fourSamplePoints:
                    samplePoint.append(sampleImageColor(image,samplePoint[0],samplePoint[1]))

                samplePoints[-1].append(fourSamplePoints)
        return samplePoints



    def getNearestPoint(self,x,y):
        nearestPointRow=-1
        nearestPointCol=-1
        nearestPointDist=100000000000
        nearestPoint=[];
        for (rowI,row) in enumerate(self.getGrid()):
            for (colI, point) in enumerate(row):
                distance=dist([x,y],point)
                if(distance<nearestPointDist):
                    nearestPointDist=distance
                    nearestPointRow=rowI
                    nearestPointCol=colI
                    nearestPoint=point
        return {"row":nearestPointRow, "col":nearestPointCol,"point":nearestPoint}
    def getNearestFixedPoint(self,x,y):
        nearestPoint=self.fixedPoints[0];
        nearestPointDist=100000000000
        for point in self.fixedPoints:
            distance=dist([x,y],point["node"].xyAsArray())
            if(distance<nearestPointDist):
                nearestPointDist=distance
                nearestPoint=point
        return nearestPoint

    def selectNearestFixedPoint(self,x,y):
        self.selectedPoint=self.getNearestFixedPoint(x,y)

    def splitAtClosestPoint(self,x,y):
        a=self.getNearestPoint(x,y)
        self.addFixedPointRecursive(Node(a["point"][0], a["point"][1]), a["row"], a["col"])

    def getPointPosition(self,row,col):
        rowStart=getIntermediate(self.topLeft.xyAsArray(), self.bottomLeft.xyAsArray(), row/(self.rows-1))
        rowEnd=getIntermediate(self.topRight.xyAsArray(), self.bottomRight.xyAsArray(), row/(self.rows-1))
        pos=getIntermediate(rowStart,rowEnd,col/(self.cols-1))

        #MAKE THIS WORK
        return self.getGrid()[row][col]
        return pos

    def addFixedPointNotRecursive(self,node,row,col):
        for fixedPoint in self.fixedPoints:
            if(fixedPoint["row"]==row and fixedPoint["col"]==col):
                return;


        if row not in self.fixedRows:
            self.fixedRows.append(row)
        if col not in self.fixedCols:
            self.fixedCols.append(col)
        self.fixedPoints.append({"row":row, "col":col, "node":node})

    def addFixedPointRecursive(self,node,row,col):

        toAdd=[(node,row,col)]


        for rowIndex in self.fixedRows:
            xy=self.getPointPosition(rowIndex,col)
            node=Node(xy[0],xy[1])
            toAdd.append((node,rowIndex,col))
        for colIndex in  self.fixedCols:
            xy=self.getPointPosition(row,colIndex)
            node=Node(xy[0],xy[1])
            toAdd.append((node,row,colIndex))

        #we have to add them all at the end like this so that the getPointPosition function doesn't get messed up by an invalid grid
        for i in toAdd:
            self.addFixedPointNotRecursive(*i)


    def getSortedFixedPoints(self):
        def eval(node):
            return node["row"]*2*self.cols+node["col"]
        self.fixedPoints.sort(key=eval)
        return self.fixedPoints
    def updateDragging(self,x,y):
        if(self.dragging):
            self.selectedPoint["node"].x=x;
            self.selectedPoint["node"].y=y

    def drawData(self, im):
        if not self.dragging:
            samplePoints=self.getSamplePoints()

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
    def dataAsString(self):
        string=""
        for (rowI, row) in enumerate(self.getSamplePoints()):
            for (vertexI, vertex) in enumerate(row):
                for (pointI, point) in enumerate(vertex):
                    string+=str(point[2])+", "
                string+="\t"
            string+="\n"
        return string




n=NodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),40,35)




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
