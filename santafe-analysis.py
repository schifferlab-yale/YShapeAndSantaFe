import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
import csv

#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)



parser = argparse.ArgumentParser(description='Santa fe csv analysis')
parser.add_argument("file", type=str, help="Path to csv file")
args=parser.parse_args()

try:

    file=open(args.file,newline="\n")
except:
    raise Exception("Error with file")

csvreader= csv.reader(file, delimiter=",")
if next(csvreader)[0] == "first row horizontal":
    firstRowHorizontal=True
else:
    firstRowHorizontal=False

#CONSTANTS FOR VALUES IN THE DATA ARRAY
BAD_DATA=0
UPLEFT_ARROW=-1
DOWNRIGHT_ARROW=1
DIMER=2
PERIPHERAL_CENTER=3
INTERIOR_CENTER=4

#load the data from the csv into the data array
data=[]
for row in csvreader:
    data.append([])
    for char in row:
        if(char==" "):
            data[len(data)-1].append(None)
        elif(char==""):
            pass
        else:
            data[len(data)-1].append(int(char))




#keeps track of lines and can also group them into connecting larger shapes
class LineMesh:

    def __init__(self,rows, cols):
        self.rows=rows
        self.cols=cols
        self.lines=[]#lines are stored like [startRow,startCol,endRow,endCol]
        self.lineGroups=[]#stores lists of lines where all lines in the list are touching

    def addLine(self,coord1, coord2):
        #add the line
        self.lines.append(coord1+coord2)

        #now put it into the right group

        groups=[]#indices of line groups it is touching
        for (groupI,group) in enumerate(self.lineGroups):
            for line in group:
                addToGroup=False

                #DANGER: HERE I USE THE DATA ARRAY WHICH IS NOT PART OF THE CLASS WHICH MAKES THIS FUNCTION CRAP
                if(data[coord1[0]][coord1[1]]!=INTERIOR_CENTER and (line[0:2]==coord1 or line[2:4]==coord1)):
                    addToGroup=True
                if(data[coord2[0]][coord2[1]]!=INTERIOR_CENTER and (line[0:2]==coord2 or line[2:4]==coord2)):
                    addToGroup=True
                if addToGroup and groupI not in groups:
                    groups.append(groupI)


        if(len(groups)==0):#if it is not touching any line groups, make a new line group
            self.lineGroups.append([coord1+coord2])
            return
        elif(len(groups)==1):#if it is touching one line group, add it to that one
            self.lineGroups[groups[0]].append(coord1+coord2)
        else: #touching multiple line groups which need to be combined
            newGroup=[]#combination of toching line groups
            for groupI in groups:
                newGroup+=self.lineGroups[groupI]#add group to new group
                self.lineGroups[groupI]=None#set it to None instead of delete so theres no indexing issues
            self.lineGroups=[lineGroup for lineGroup in self.lineGroups if lineGroup is not None]#now delete the old group
            newGroup.append(coord1+coord2)#add the line to the new, larger group
            self.lineGroups.append(newGroup)#add the larger group back into the list of line groups

    #takes a line group and finds their "center of mass"
    def getCenterOfMass(self,lineGroup):
        totalWeight=0#total length of all the lines
        centerX=0#weighted sum of the coordinates of the center of each line
        centerY=0
        for line in lineGroup:
            weight=math.sqrt(pow(line[0]-line[2],2)+pow(line[1]-line[3],2))#weight=length
            totalWeight+=weight
            centerX+=(line[0]+line[2])/2*weight#add the center of the line (average of the start and end)
            centerY+=(line[1]+line[3])/2*weight

        #divide be total weight to get center of mass
        centerX/=totalWeight
        centerY/=totalWeight
        return [centerX,centerY]

    #finds the two furthest apart points in the line group
    def getFurthestPoints(self,lineGroup):
        #get all the points in that line group
        points=self.lineGroupToPoints(lineGroup)

        furthestDist=0
        pointA=points[0]
        pointB=points[0]
        for point1 in points:
            for point2 in points:#compare every point with every other point
                dist=math.sqrt(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2))
                if(dist>furthestDist):
                    furthestDist=dist
                    pointA=point1
                    pointB=point2
        return pointA,pointB

    #simply take a line group and return all the start and end points of lines
    def lineGroupToPoints(self,lineGroup):
        points=[]
        for line in lineGroup:
            if line[0:2] not in points:
                points.append(line[0:2])
            if line[2:4] not in points:
                points.append(line[2:4])
        return points

    #given a line group get the convex hull
    def getConvexHull(self,lineGroup):
        #to understand algorithm go here
        #https://startupnextdoor.com/computing-convex-hull-in-python/
        points=self.lineGroupToPoints(lineGroup)
        hull_points=[]

        # get leftmost point
        start = points[0]
        min_x = start[0]
        for p in points[1:]:
            if p[0] < min_x:
                min_x = p[0]
                start = p

        point = start
        hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction =(((p2[0] - point[0]) * (far_point[1] - point[1]))- ((far_point[0] - point[0]) * (p2[1] - point[1])))
                    if direction > 0:
                        far_point = p2

            hull_points.append(far_point)
            point = far_point
        return hull_points

    #given a specified row and column, calulate where it would be drawn on the output image
    def rowColToDrawCoord(self,rowCol,image):
        padding=50
        imageWidth=image.shape[0]#assume it is square

        y=padding+rowCol[0]/self.rows*(imageWidth-2*padding)
        x=padding+rowCol[1]/self.cols*(imageWidth-2*padding)
        return (int(x),int(y))

    #draws the lines onto an output image
    def draw(self,outputImage):
        np.random.seed(0)#set the seed so we get consistent colors between runs and drawings
        for group in self.getLineGroupsWithAtLeastTwoInteriors():
            color = list(np.random.random(size=3) * 256)#get random color
            for line in group:
                #draw line in that group
                start=self.rowColToDrawCoord(line[0:2],outputImage)
                end=self.rowColToDrawCoord(line[2:4],outputImage)
                outputImage=cv2.line(outputImage,start,end,color,2)

    #draws the calculated vectors or convex hulls onto an output image
    def drawVectors(self,outputImage):
        np.random.seed(0)#set the seed so we get consistent colors between runs and drawings
        for group in self.getLineGroupsWithAtLeastTwoInteriors():
            color = list(np.random.random(size=3) * 256)#random color


            COM=self.getCenterOfMass(group)
            point=self.rowColToDrawCoord(COM,outputImage)
            outputImage=cv2.circle(outputImage,point,4,color,-1)

            pointA,pointB = self.getFurthestPoints(group)#two ends of the vector

            vector=[pointA[0]-pointB[0],pointA[1]-pointB[1]]

            #draw vector centered at COM
            start=self.rowColToDrawCoord([COM[0]-vector[0]/2, (COM[1]-vector[1]/2)],outputImage)
            end=self.rowColToDrawCoord([COM[0]+vector[0]/2, (COM[1]+vector[1]/2)],outputImage)

            outputImage=cv2.line(outputImage,start,end,color,8)

            #draw the convex hull too
            hull=self.getConvexHull(group)
            for i in range(len(hull)):#loop around and draw each line to the next
                start=hull[i]
                end=hull[(i+1)%len(hull)]

    def getLineGroupsWithAtLeastTwoInteriors(self):
        output=[]
        for group in self.lineGroups:
            interiorCount=0
            for line in group:
                if(data[line[0]][line[1]]==INTERIOR_CENTER or data[line[2]][line[3]]==INTERIOR_CENTER):
                    interiorCount+=1
            if(interiorCount>=2):
                output.append(group)
        return output
    #returns array in format of [locationX, locationY, vectorX,vectorY]
    def getLineGroupVectorsAndLocations(self):
        vectorLocations=[]
        for group in self.lineGroups:
            COM=self.getCenterOfMass(group)
            pointA,pointB = self.getFurthestPoints(group)#two ends of the vector
            thisEntry=[COM[0],COM[1],pointA[0]-pointB[0],pointA[1]-pointB[1]]
            vectorLocations.append(thisEntry)
        return vectorLocations





#just shifts every element in an array by one and loops the last one back to the front
def cyclicRotate(input):
     return ([input[-1]] + input[0:-1])

#inverts all numbers in an array and keeps None values
def invert(input):
    return [-i if i is not None else None for i in input]

#gets a given point in the data array, returns none if out of bounds
def getPoint(row,col):
    if(row>=0 and row<len(data) and col>=0 and col<len(data[row])):
        return data[row][col]
    else:
        return None

#if the nth row has horizontal arrows or vertial arrows
def isRowHorizontal(rowI):
    return firstRowHorizontal and rowI%2==0 or not firstRowHorizontal and rowI%2==1

#get the values of the four surrounding points to a given row,col in data
def getNeighbors(rowI,colI):

    #get four neigboring islands
    if(isRowHorizontal(rowI)):#if row is vertical, nothing points into it from the top or bottom
        top=getPoint(rowI-1,colI)
        bottom=getPoint(rowI+1,colI)
    else:
        top=None
        bottom=None

    left=getPoint(rowI,colI-1)
    right=getPoint(rowI,colI+1)

    #Flip these so -1 means out and 1 means in
    if right is not None:
        right=right*-1
    if bottom is not None:
        bottom=bottom*-1

    return top,right,bottom,left

#same as getNeighbors but unless the neighbor is an arrow it will return NONE
def getArrowNeighbors(rowI,colI):
    top,bottom,left,right=getNeighbors(rowI,colI)
    if(top!=UPLEFT_ARROW and top!=DOWNRIGHT_ARROW):
        top=None
    if(bottom!=UPLEFT_ARROW and bottom!=DOWNRIGHT_ARROW):
        bottom=None
    if(left!=UPLEFT_ARROW and left!=DOWNRIGHT_ARROW):
        left=None
    if(right!=UPLEFT_ARROW and right!=DOWNRIGHT_ARROW):
        right=None
    return top, bottom, left, right

#given a center of a rectangle(plaquetts), get the six surrounding verticies
def getRectangleEdgeCoords(centerRow, centerCol):
    if(isRowHorizontal(centerRow)):
        return [
            [centerRow-2, centerCol-1],
            [centerRow-2, centerCol+1],
            [centerRow, centerCol-1],
            [centerRow, centerCol+1],
            [centerRow+2, centerCol-1],
            [centerRow+2, centerCol+1]
        ]
    else:
        return [
            [centerRow-1, centerCol-2],
            [centerRow-1, centerCol],
            [centerRow-1, centerCol+2],
            [centerRow+1, centerCol-2],
            [centerRow+1, centerCol],
            [centerRow+1, centerCol+2],
        ]

#gets the number of arrow islands around a given row,col
def countNeighboringIslands(rowI,colI):
    #count surrounding islands
    top,right,bottom,left=getNeighbors(rowI,colI)
    allVerticies=[top,right,bottom,left]
    islandCount=0
    for v in allVerticies:
        if v==UPLEFT_ARROW or v==DOWNRIGHT_ARROW: islandCount+=1
    return islandCount

#calculates whether a vertex is a dimer or not based on verticies array
def getVertexType(rowI,colI):
    top,right,bottom,left=getArrowNeighbors(rowI,colI)
    allVerticies=[top,right,bottom,left]

    islandCount=countNeighboringIslands(rowI,colI)

    vertexType=0;

    if islandCount>1 and cell is None:
        #get vertex type
        for type in verticies[islandCount].keys():
            if allVerticies in verticies[islandCount][type]:
                vertexType=type
                break
    return vertexType

def drawArrowCell(rowI,cellI,outputImage):#just draws an arrow at the specified coords
    #draw verticies
    if cell is not None:
        if(isRowHorizontal(rowI)):
            if(cell==DOWNRIGHT_ARROW):
                outputImage=cv2.arrowedLine(outputImage,(int(x-arrowLength),int(y)),(int(x+arrowLength),int(y)), 2, tipLength=0.3)
            elif(cell==UPLEFT_ARROW):
                outputImage=cv2.arrowedLine(outputImage,(int(x+arrowLength),int(y)),(int(x-arrowLength),int(y)), 2, tipLength=0.3)

        else:
            if(cell==DOWNRIGHT_ARROW):
                outputImage=cv2.arrowedLine(outputImage,(int(x),int(y-arrowLength)),(int(x),int(y+arrowLength)), 2, tipLength=0.3)
            elif(cell==UPLEFT_ARROW):
                outputImage=cv2.arrowedLine(outputImage,(int(x),int(y+arrowLength)),(int(x),int(y-arrowLength)), 2, tipLength=0.3)

def drawCell(rowI,cellI,x,y,outputImage):#draw cell based on whatevere type it is
    cell=getPoint(rowI,cellI)
    horizontal=isRowHorizontal(rowI)
    if cell == DOWNRIGHT_ARROW:
        if(horizontal):
            outputImage=cv2.arrowedLine(outputImage,(int(x-arrowLength),int(y)),(int(x+arrowLength),int(y)), 2, tipLength=0.3)
        else:
            outputImage=cv2.arrowedLine(outputImage,(int(x),int(y-arrowLength)),(int(x),int(y+arrowLength)), 2, tipLength=0.3)
    elif cell == UPLEFT_ARROW:
        if(horizontal):
            outputImage=cv2.arrowedLine(outputImage,(int(x+arrowLength),int(y)),(int(x-arrowLength),int(y)), 2, tipLength=0.3)
        else:
            outputImage=cv2.arrowedLine(outputImage,(int(x),int(y+arrowLength)),(int(x),int(y-arrowLength)), 2, tipLength=0.3)
    elif cell==DIMER:
        outputImage=cv2.circle(outputImage,(int(x),int(y)), 4,RED,-1)
    elif cell==PERIPHERAL_CENTER:
        pass
    elif cell==INTERIOR_CENTER:
        outputImage=cv2.circle(outputImage,(int(x),int(y)), 4,BLUE,-1)



def getNearbyCenters(rowI,colI):#gets the centers which a given coordinant is a vertex of
    centers=[]
    for rowOff in range(-2,3):
        for colOff in range(-2,3):
            if(getPoint(rowI+rowOff,colI+colOff)==PERIPHERAL_CENTER or getPoint(rowI+rowOff,colI+colOff)==INTERIOR_CENTER):
                centers.append([rowI+rowOff,colI+colOff])
    return centers

#given a row column, returns an array of the coordinates of centers it connects to
def getStringConnections(rowI,colI):
    horizontal=isRowHorizontal(rowI)
    islandCount=countNeighboringIslands(rowI,colI)
    top,right,bottom,left = getNeighbors(rowI,colI)

    #get the 3 or four centers it could connect to
    centers=getNearbyCenters(rowI,colI)

    connections=[]

    for i in range(len(centers)):
        center=centers[i]
        relativeRow=center[0]-rowI
        relativeCol=center[1]-colI

        if(relativeRow>0 and relativeCol>0): #bottom right
            if(bottom == right):
                connections.append(center)
        elif(relativeRow<0 and relativeCol>0): #top right
            if(top==right):
                connections.append(center)
        elif(relativeRow<0 and relativeCol<0):# top left
            if(top==left):
                connections.append(center)
        elif(relativeRow>0 and relativeCol<0):#bottom left
            if(bottom==left):
                connections.append(center)

        elif(relativeRow==0 ):#right or left
            if(islandCount==2 and top== bottom or islandCount==3 and top!=bottom):
                connections.append(center)
        elif(relativeCol==0):#top or bottom
            if(islandCount==2 and right==left or islandCount==3 and right!=left):
                connections.append(center)


    return connections


#basic shapes for each vertex count and type (does not include rotations or inversionsn)
baseVerticies={
    2:{
        1:[[None,1,None,-1],[1,-1,None,None]], #type 1
        2:[[1,None,1,None],[1,1,None,None]], #type 2
    },

    3:{
        1:[[-1,1,-1,None]], #type 1
        2:[[1,1,-1,None],[1,-1,-1,None]], #type 2
        3:[[1,1,1,None]], #type 3
    },

    4:{
        1:[[1,-1,1,-1]],
        2:[[1,-1,-1,1]],
        3:[[1,-1,1,1]],
        4:[[1,1,1,1]]
    }
}



#format: verticies[number of islands][type][arrangement]
#calculate all possible rotations and inversions of the base verticies and stores that in the verticies dictionary
verticies={}
for islandCount in [2,3,4]:
    verticies[islandCount]={}
    for type in baseVerticies[islandCount].keys():
        verticies[islandCount][type]=[]
        for combination in baseVerticies[islandCount][type]:
            for i in range(4):
                combination=cyclicRotate(combination)
                if ( combination not in verticies[islandCount][type]):
                    verticies[islandCount][type].append(combination)
                if( invert(combination) not in verticies[islandCount][type]):
                    verticies[islandCount][type].append(invert(combination))




#calculates all other values in data besides those given in array
for (rowI, row) in enumerate(data):
    horizontal= firstRowHorizontal and rowI%2==0 or not firstRowHorizontal and rowI%2==1
    for(colI, cell) in enumerate(row):
        top,right,bottom,left=getNeighbors(rowI,colI)
        allVerticies=[top,right,bottom,left]
        islandCount=countNeighboringIslands(rowI,colI)
        vertexType=getVertexType(rowI,colI)



        #draw dimers
        if horizontal:
            if vertexType>1:
                data[rowI][colI]=DIMER


        #calculate center of rectangles
        center=False
        if(horizontal):
            center=True
        if(islandCount==0 and cell is None):
            data[rowI][colI]=PERIPHERAL_CENTER
            if horizontal and getPoint(rowI,colI-2)==PERIPHERAL_CENTER:
                data[rowI][colI]=INTERIOR_CENTER
                data[rowI][colI-2]=INTERIOR_CENTER
            elif not horizontal and getPoint(rowI-2,colI)==PERIPHERAL_CENTER:
                data[rowI][colI]=INTERIOR_CENTER
                data[rowI-2][colI]=INTERIOR_CENTER


lines=LineMesh(len(data),len(data[0]))

imageWidth=1000
padding=50

outputImage=np.zeros((imageWidth,imageWidth,3), np.uint8)
outputImage[:,:]=(250,250,250)

spacingX=(imageWidth-2*padding)/(len(data[0])-1)
spacingY=(imageWidth-2*padding)/(len(data)-1)

arrowLength=spacingX/2

#make the line mesh
for (rowI, row) in enumerate(data):
    y=padding+rowI/len(data)*(imageWidth-2*padding)
    for(colI, cell) in enumerate(row):
        x=padding+colI/len(data[rowI])*(imageWidth-2*padding)
        drawCell(rowI,colI,x,y,outputImage)

        if(cell==DIMER):
            connections=getStringConnections(rowI,colI)
            for center in connections:
                lines.addLine([rowI,colI],center)

                #startX=int(x)
                #startY=int(y)
                #endX=int(padding+center[1]/len(data[rowI])*(imageWidth-2*padding))
                #endY=int(padding+center[0]/len(data)*(imageWidth-2*padding))
                #outputImage=cv2.line(outputImage,(startX,startY),(endX,endY),RED,2)






lines.draw(outputImage)

vectorOutputImage=np.zeros((imageWidth,imageWidth,3), np.uint8)
vectorOutputImage[:,:]=(250,250,250)
lines.drawVectors(vectorOutputImage)
#lines.drawVectors(outputImage)

cv2.imshow("window",outputImage)
cv2.imshow("window2",vectorOutputImage)
cv2.waitKey(0)
cv2.imwrite("analysis-output.jpg", np.float32(outputImage));
cv2.imwrite("analysis-output-vectors.jpg", np.float32(vectorOutputImage));
