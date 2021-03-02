import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse

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

#takes in an [x1,y1] and [x2,y2] and returns their distance
def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))
#takes in an [x1,y1] and [x2,y2] and returns a point a certain percent between the two
def getIntermediate(point1,point2,percent):
    return [(point2[0]-point1[0])*percent+point1[0], (point2[1]-point1[1])*percent+point1[1]]


#Gets the color of an area on the screen at (x,y)
#It will look at every pixel within a certain with of the specified point and then
#average their values
def sampleImageColor(im,x,y):
    avg=0;
    count=0;#number of pixels checked
    width=2;#distance away from center pixel to sample
    x=int(x)
    y=int(y)

    imWidth, imHeight, channels=im.shape

    #loop through rows and columns
    for i in range(x-width,x+width):
        for j in range(y-width,y+width):
            #make sure pixel is not offscreen
            if i<0 or j<0 or i>imWidth-1 or j>imHeight-1:
                continue
            #add to average
            avg+=im[j][i][0]#(im[j][i][0]+im[j][i][1]+im[j][i][2])/3
            count+=1

    #prevent divide by 0 error
    if count==0:
        return 0

    #return avg color
    avg/=count;
    if(avg>127):
        return 1
    return -1

#Basic class to hold an x and y value
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

#turns [x,y] into (x,y)
def xyArrayToIntTuple(arr):
    return (int(arr[0]),int(arr[1]))


#main class
class NodeNetwork:
    def __init__(self,topLeft,topRight,bottomLeft,bottomRight,rows, cols):
        #number of rows and columns
        self.rows=rows;
        self.cols=cols;

        #variables to help with point dragging
        self.selectedPoint=None;
        self.dragging=False

        #format {row: rowIndex, col:colIndex, node:Node}
        self.fixedPoints=[
        {"row":0,"col":0,"node":topLeft},
        {"row":0,"col":cols-1,"node":topRight},
        {"row":rows-1,"col":0,"node":bottomLeft},
        {"row":rows-1, "col":cols-1, "node":bottomRight}
            ]

        #defines boundaries
        self.topLeft=self.fixedPoints[0]["node"];
        self.topRight=self.fixedPoints[1]["node"];
        self.bottomLeft=self.fixedPoints[2]["node"];
        self.bottomRight=self.fixedPoints[3]["node"];

        #rows and columns which contain fixed points
        self.fixedRows=[0,rows-1]
        self.fixedCols=[0,cols-1]



    def draw(self,im):

        #draw all fixed points
        sortedPoints=self.getSortedFixedPoints()
        for (i,point) in enumerate(self.getSortedFixedPoints()):
            #draw fixed point
            im=cv2.circle(im,point["node"].xyAsIntTuple(), 5,RED,-1)

            #hasGrid represents if the point is the topLeft corner of a subgrid
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
                #if it has a valid grid then get the coordinants of the verticies
                topRight=sortedPoints[i+1]
                bottomLeft=sortedPoints[i+len(self.fixedCols)]
                bottomRight=sortedPoints[i+len(self.fixedCols)+1]
                cols=topRight["col"]-topLeft["col"]
                rows=bottomLeft["row"]-topLeft["row"]

                topRightCoord=topRight["node"].xyAsArray()
                bottomLeftCoord=bottomLeft["node"].xyAsArray()
                bottomRightCoord=bottomRight["node"].xyAsArray()

                #draw the outline and grid
                im=cv2.line(im,xyArrayToIntTuple(topLeftCoord),xyArrayToIntTuple(topRightCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(topLeftCoord),xyArrayToIntTuple(bottomLeftCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(bottomLeftCoord),xyArrayToIntTuple(bottomRightCoord),BLACK,1)
                im=cv2.line(im,xyArrayToIntTuple(topRightCoord),xyArrayToIntTuple(bottomRightCoord),BLACK,1)

                for (startX,startY,endX,endY) in zip(np.linspace(topLeftCoord[0],topRightCoord[0],cols+1), np.linspace(topLeftCoord[1],topRightCoord[1],cols+1), np.linspace(bottomLeftCoord[0],bottomRightCoord[0],cols+1), np.linspace(bottomLeftCoord[1],bottomRightCoord[1],cols+1)):
                    im=cv2.line(im,(int(startX),int(startY)),(int(endX),int(endY)),RED,1)
                for (startX,startY,endX,endY) in zip(np.linspace(topLeftCoord[0],bottomLeftCoord[0],rows+1), np.linspace(topLeftCoord[1],bottomLeftCoord[1],rows+1), np.linspace(topRightCoord[0],bottomRightCoord[0],rows+1), np.linspace(topRightCoord[1],bottomRightCoord[1],rows+1)):
                    im=cv2.line(im,(int(startX),int(startY)),(int(endX),int(endY)),RED,1)

        #draw the sample points if we are not dragging
        if not self.dragging:
            samplePoints=self.getSamplePoints()
            for (rowI, row) in enumerate(samplePoints):
                for (vertexI, vertex) in enumerate(row):
                    for (pointI, point) in enumerate(vertex):
                        #Draw Point based on color
                        if(point[2]==1):
                            im=cv2.circle(im,(int(point[0]),int(point[1])),2,RED,-1)
                        else:
                            im=cv2.circle(im,(int(point[0]),int(point[1])),2,BLUE,-1)

                        #Check for errors:

                    #ERROR DETECTION
                    #if we are not in the last column, make sure the right point is opposite color than the one next to it
                    if vertexI < len(row)-1:
                        if(vertex[2][2] == row[vertexI+1][1][2]):
                            im=cv2.circle(im,(int(vertex[2][0]),int(vertex[2][1])),5,GREEN,2)
                    #if we are not in the last row, make sure bottom ponit is the opposite color than the one below it
                    if rowI < len(samplePoints)-1:
                        if(vertex[3][2] == samplePoints[rowI+1][vertexI][0][2]):
                            im=cv2.circle(im,(int(vertex[3][0]),int(vertex[3][1])),5,GREEN,2)

    #generates the grid based on the subgrids
    def getGrid(self):
        #generate an empty grid representing each point
        grid=[]
        for row in range(self.rows):
            grid.append([])
            for col in range(self.cols):
                grid[len(grid)-1].append("")

        #loop through each fixed point and draw the grid that it is the top left of
        sortedPoints=self.getSortedFixedPoints()
        for (i,point) in enumerate(self.getSortedFixedPoints()):

            #make sure that point is the topleft ofo a grid and not an edge
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
                #get grid coordinates if valid grid
                topRight=sortedPoints[i+1]
                bottomLeft=sortedPoints[i+len(self.fixedCols)]
                bottomRight=sortedPoints[i+len(self.fixedCols)+1]
                cols=topRight["col"]-topLeft["col"]
                rows=bottomLeft["row"]-topLeft["row"]

                topRightCoord=topRight["node"].xyAsArray()
                bottomLeftCoord=bottomLeft["node"].xyAsArray()
                bottomRightCoord=bottomRight["node"].xyAsArray()

            #get all the points in that subgrid
            if(hasGrid):
                for (rowI, rowStartX,rowStartY,rowEndX,rowEndY) in  zip(range(0,rows+1), np.linspace(topLeftCoord[0],bottomLeftCoord[0],rows+1), np.linspace(topLeftCoord[1],bottomLeftCoord[1],rows+1), np.linspace(topRightCoord[0],bottomRightCoord[0],rows+1), np.linspace(topRightCoord[1],bottomRightCoord[1],rows+1)):
                    for(colI, pointX, pointY) in zip(range(0,cols+1), np.linspace(rowStartX,rowEndX,cols+1), np.linspace(rowStartY,rowEndY,cols+1)):
                        grid[point["row"]+rowI][point["col"]+colI]=[pointX,pointY]
        return grid


    #get sample points as [row][vertex][island] which is an array [x,y,color]
    def getSamplePoints(self):
        #multiplier for how far the sample points are from the edge of the square
        shiftConstant=0.25
        grid=self.getGrid()
        samplePoints=[]
        for (rowI, row) in enumerate(grid[:-1]):
            samplePoints.append([])
            for (pointI,point) in enumerate(row[:-1]):

                #get corners of square
                topLeft=point
                topRight=grid[rowI][pointI+1]
                bottomLeft=grid[rowI+1][pointI]
                bottomRight=grid[rowI+1][pointI+1]

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
                #get what color each point is
                for samplePoint in fourSamplePoints:
                    samplePoint.append(sampleImageColor(image,samplePoint[0],samplePoint[1]))

                samplePoints[-1].append(fourSamplePoints)
        return samplePoints


    #gets the nearest point on the grid to (x,y)
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

    #gets the nearest fix point to (x,y)
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

    #splits gridi into four at the closest ponit
    def splitAtClosestPoint(self,x,y):
        a=self.getNearestPoint(x,y)
        self.addFixedPointRecursive(Node(a["point"][0], a["point"][1]), a["row"], a["col"])

    #get the position of a point at given row and column
    def getPointPosition(self,row,col):
        return self.getGrid()[row][col]

    #adds a fixed point to a row and col
    #BE CAREFUL IT WILL NOT CHECK IF OTHER POINTS NEED TO BE ADDED TO MAKE EVERYTHING SQUARE
    def addFixedPointNotRecursive(self,node,row,col):
        for fixedPoint in self.fixedPoints:
            if(fixedPoint["row"]==row and fixedPoint["col"]==col):
                return;


        if row not in self.fixedRows:
            self.fixedRows.append(row)
        if col not in self.fixedCols:
            self.fixedCols.append(col)
        self.fixedPoints.append({"row":row, "col":col, "node":node})

    #safe to use function to make a given row and col location a fixed ponit
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

    #git fixed points sorted left-to-right, top-to-bottom
    def getSortedFixedPoints(self):
        def eval(node):
            return node["row"]*2*self.cols+node["col"]
        self.fixedPoints.sort(key=eval)
        return self.fixedPoints

    #drag to x,y
    def updateDragging(self,x,y):
        if(self.dragging):
            self.selectedPoint["node"].x=x;
            self.selectedPoint["node"].y=y

    #draw output
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
    def dataAsString(self):
        string=""
        for (rowI, row) in enumerate(self.getSamplePoints()):
            for (vertexI, vertex) in enumerate(row):
                for (pointI, point) in enumerate(vertex):
                    string+=str(point[2])+", "
                string+="\t"
            string+="\n"
        return string




n=NodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),args.rows, args.columns)




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
