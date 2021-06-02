import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
from random import shuffle
import argparse
import csv





#constants
#cv2 stores colors as BGR, not RGB
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)


#a simple class to store a vertex energy based by counting paralell and perpendicular interations
class VertexEnergy:
    def __init__(self,par,perp):
        self.par=par
        self.perp=perp
    def __repr__(self):
        return f"{self.par}|| + {self.perp}x"
    
    def __add__(self,other):
        return VertexEnergy(self.par+other.par,self.perp+other.perp)
    
    def __eq__(self,other):
        return other.par==self.par and other.perp == self.perp
    
    def compare(self,other):
        if(self==other):
            return 0
        if(self.par>other.par and self.perp>=other.perp or self.par>= other.par and self.perp>other.perp):
            return 1
        if(other.par>self.par and other.perp>=self.perp or other.par>=self.par and other.perp>self.perp):
            return -1
        if(other.par>self.par and other.perp<self.perp or other.par<self.par and other.perp>self.perp):
            return None #can't tell which one is bigger
        raise Exception("all cases should be caught")


#just shifts every element in an array by one and loops the last one back to the front
#eg cyclicRotate([a,b,c,d])=[d,a,b,c]
def cyclicRotate(input):
     return ([input[-1]] + input[0:-1])

#inverts all numbers in an array and keeps None values
#e.g. invert([1,2,None,-3,None,-4])=[-1,-2,None,3,None,4]
def invert(input):
    return [-i if i is not None else None for i in input]

#takes two points of the form (x,y) and returns their distance on a plane
#e.g. dist((1,1),(4,5))=5
def dist(point1,point2):
    return math.sqrt(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2))

#takes an array of point tuples and returns the two points which are furthest away
# e.g. getFurthestPoints([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])=[(x1,y1),(x3,y3)]
def getFurthestPoints(points):
    if(len(points)<2):
        raise("must have at least two points")

    furthestPoints=[None,None]#
    furthestDist=-1

    for point1 in points:
        for point2 in points:
            distance=dist(point1,point2)
            if(distance>furthestDist):
                furthestDist=distance
                furthestPoints=[point1,point2]
    return furthestPoints

#works similar to get getFurthestPoints() except it will return an array with all pairs of points which are the furtheest distance apart
def getAllFurthestPoints(points):
    if(len(points)<2):
        raise("must have at least two points")

    furthestPoints=[]
    furthestDist=-1

    for point1 in points:
        for point2 in points:
            distance=dist(point1,point2)
            if(abs(distance-furthestDist)<0.00001):
                furthestPoints.append([point1,point2])
            elif(distance>furthestDist):
                furthestDist=distance
                furthestPoints=[[point1,point2]]
    return furthestPoints

#returns the center of a group of points
def avgPoint(points):
    avgX=0.0;
    avgY=0.0;
    for point in points:
        avgX+=point[0]
        avgY+=point[1]
    avgX/=len(points)
    avgY/=len(points)
    return (avgX,avgY)


#does a linear regression through a group of points and returns their angle
def getAngle(points):
    avg=avgPoint(points)

    numerator=0.0
    denominator=0.0
    Sx,Sy,Sxx,Syy,Sxy=0,0,0,0,0
    n=len(points)
    for point in points:
        Sx+=point[0]
        Sy+=point[1]
        Sxy+=point[0]*point[1]
        Sxx+=point[0]*point[0]
        Syy+=point[1]*point[1]
        numerator+=(point[0]-avg[0])*(point[1]-avg[1])
        denominator+=(point[0]-avg[0])*(point[0]-avg[0])

    numerator=(n*Sxy-Sx*Sy)
    denominator=(n*Sxx-Sx*Sx)

    if denominator==0:
        return math.pi/2
    else:
        return math.atan(numerator/denominator)

#does a linear regression through a group of points and returns the y-intercept
def getIntercept(points):
    avg=avgPoint(points)
    slope=getSlope(points)

    return avg[1]-slope*avg[0]

#does a linear regression on a group of points and returns the angle of its slope
def getSlope(points):
    avg=avgPoint(points)
    numerator=0
    denominator=0
    for point in points:
        numerator+=(point[0]-avg[0])*(point[1]-avg[1])
        denominator+=(point[0]-avg[0])*(point[0]-avg[0])

    if(denominator==0):
        return 0
    return numerator/denominator

#takes a graph of the form [[a,b,c],[d,e,f],...] and returns if there is a cycle
#the input representation is that each index in the list is a node and each element in the
#list at that index is the nodes it connects
def graphHasCycle(graph):
    return _graphHasCycle(graph,[False]*len(graph),0,-1)

#main function for the grahp has cycle.
#it works by traversing down the nodes and recursing at each split. If it ever sees a previous node then
#there is a loop
def _graphHasCycle(graph,seenPoints,point,prevPoint):
    seenPoints=[i for i in seenPoints]#deep copy
    seenPoints[point]=True
    for nextPoint in graph[point]:
        if(nextPoint != prevPoint):
            if(seenPoints[nextPoint]):
                return True
            else:
                if(_graphHasCycle(graph, seenPoints, nextPoint, point)):
                    return True
    return False

#gets the distance between num1 and num2 assuming they range from -pi/2 to pi/2 and wrap over
#eg piDist(0,1)=1, piDist(-pi/2+0.1, pi/2-0.1)=0.2
def piDist(num1,num2):

    if(num1>math.pi/2 or num2>math.pi/2 or num1<-math.pi/2 or num2<-math.pi/2):
        raise("must be from -pi/2 to pi/2")

    dist1=abs(num1-num2)
    dist2=abs(num1-num2+math.pi)
    dist3=abs(num1-num2-math.pi)
    return min([dist1,dist2,dist3])

#a class which holds two points and has functions that treat those two points like the
#start and end of a line segment
class LineSegment():
    def __init__(self, startX,startY,endX,endY):
        self.startX=startX
        self.startY=startY
        self.endX=endX
        self.endY=endY

    #takes another line segment and checks whether their starts/ends have a common point
    def connectsWith(self,lineSegment):
        if(self.start==lineSegment.start or self.end==lineSegment.start or self.start==lineSegment.end or self.end==lineSegment.end):
            return True
        return False

    #takes another line segment and returns their common start/end if they have one
    def getSharedPoint(self,lineSegment):
        if(self.start==lineSegment.start or self.start==lineSegment.end):
            return self.start
        if(self.end==lineSegment.start or self.end==lineSegment.end):
            return self.end
        return None

    #returns the "dot product" of two line segments (their lengths times the cosine of their angle)
    def dotP(self, lineSegment):
        return self.startX*lineSegment.startX+self.endY*lineSegment.endY

    def asVector(self):
        return (self.endX-self.startX,self.endY-self.startY)

    def getCorrelation(self,lineSegment):
        return abs(math.cos(self.angle-lineSegment.angle))

    #returns the center of the line segment
    @property
    def CoM(self):#center of mass
        return ((self.startX+self.endX)/2,(self.startY+self.endY)/2)

    #returns the length of the line segment
    @property
    def length(self):
        return dist(self.start,self.end)

    #these two functions return the start/end of the line segment
    @property
    def start(self):
        return (self.startX,self.startY)
    @property
    def end(self):
        return (self.endX,self.endY)

    #returns an array of the start and end points
    @property
    def points(self):
        return [self.start,self.end]

    #returns the angle off of the +x axis
    @property
    def angle(self):
        if(self.endX==self.startX):#prevent divide by 0
            return math.pi/2
        return math.atan((self.endY-self.startY)/(self.endX-self.startX))

    def __eq__(self, o):
        if((self.start==o.start and self.end==o.end) or (self.start==o.end and self.end==o.start)):
            return True
        return False

#a string is essentially a collection of line segments
class String():
    def __init__(self, lineSegments):
        self.lineSegments=lineSegments
        self.color= list(np.random.random(size=3) * 256)
        self.id=np.random.randint(0,2**31-1)
        self.touchingInteriors=[]

    #adds another line segment to this string
    def addLineSegment(self,lineSegment):
        if lineSegment not in self.lineSegments:
            self.lineSegments.append(lineSegment)

    #merges two strings
    def addString(self,string):
        for line in string.lineSegments:
            self.addLineSegment(line)

    #gets all points in all line segments
    def getPoints(self):
        points=[]
        for line in self.lineSegments:
            for point in line.points:
                if not(point in points):
                    points.append(point)
        return points

    #gets a linesegment representation of the string
    def getTrace(self):
        #furthest points
        """points=self.getPoints();
        furthestPoints=getFurthestPoints(points)
        CoM=self.getCoM()

        xLength=furthestPoints[1][0]-furthestPoints[0][0]
        yLength=furthestPoints[1][1]-furthestPoints[0][1]
        return LineSegment(CoM[0]+xLength/2, CoM[1]+yLength/2, CoM[0]-xLength/2, CoM[1]-yLength/2)"""

        #average furthest point
        points=self.getPoints();
        furthestPoints=getAllFurthestPoints(points)
        avgAngle=0;
        for pair in furthestPoints:
            avgAngle+=LineSegment(*pair[0],*pair[1]).angle
        avgAngle/=len(furthestPoints)
        length=dist(*furthestPoints[0])
        CoM=self.getCoM()
        return LineSegment(CoM[0]-length/2*math.cos(avgAngle),CoM[1]-length/2*math.sin(avgAngle), CoM[0]+length/2*math.cos(avgAngle),CoM[1]+length/2*math.sin(avgAngle))

        """#linear regression
        points=self.getPoints();
        angle=getAngle(points)

        furthestPoints=getFurthestPoints(points)
        length=dist(furthestPoints[0],furthestPoints[1])
        CoM=self.getCoM()"""


        return LineSegment(CoM[0]-length/2*math.cos(angle),CoM[1]-length/2*math.sin(angle), CoM[0]+length/2*math.cos(angle),CoM[1]+length/2*math.sin(angle))



    #gets the average position of each point in the string or "center of mass"
    def getCoM(self):
        totalMass=0;
        xSum=0
        ySum=0
        for line in self.lineSegments:

            totalMass+=line.length
            CoM=line.CoM
            xSum+=CoM[0]*line.length
            ySum+=CoM[1]*line.length

        return (xSum/totalMass, ySum/totalMass)

    #returns all (row,col) points in the string with no duplicates
    def getPoints(self):
        points=[]
        for segment in self.lineSegments:
            points+=segment.points
        
        return list(set(points))#this removes duplicates

    #just sums the length of each line segement
    def getLength(self):
        length=0;
        for line in self.lineSegments:
            length+=line.length
        return length
    
    def getSegmentCount(self):
        return len(self.lineSegments)

    #returns a "graph" which is an array where each position contains indices to the other
    #other points in the array it connects to. This is used for figuring out if it is a loop 
    #or not
    def asGraph(self):
        points=self.getPoints()
        graph=[[] for point in points]

        for segment in self.lineSegments:
            startIndex=points.index(segment.start)
            endIndex=points.index(segment.end)
            graph[startIndex].append(endIndex)
            graph[endIndex].append(startIndex)

        return graph
        

    def isLoop(self):
        #create a graph of the points and check for cycles
        #https://stackoverflow.com/questions/526331/cycles-in-an-undirected-graph
        return graphHasCycle(self.asGraph())

    def __repr__(self):
        return str(self.id)


#a point in teh grid which holds information about it
class Cell:
    def __init__(self,arrow):
        assert arrow in ["up","left","down","right", None]
        self.arrow=arrow;#arrow direction or "None" if there is no arro
        self.center=False#is the center of a plaquette?
        self.dimer=False#is a dimer
        self.interiorCenter=False#is the center of an interior plaquett?
        self.compositeSquareCenter=False#is the center of a compositesquare
        self.stringsInCompositeSquare=None#how many strings are in the composite square (if it is the center of one)
        self.badData=False
    def __str__(self):
        return "cell arrow:"+str(self.arrow)


    @property
    def hasArrow(self):
        return self.arrow is not None

#main class for holding the lattice
class SantaFeLattice:
    def __init__(self,csvFile,firstCenter=None,removeEdgeStrings=True,autoAlignCenters=False, randomizeStringColor=False):

        self.randomizeStringColor=randomizeStringColor

        #csvread the data
        csvreader= csv.reader(csvFile, delimiter=",")

        #make sure we know whether the first row is horizontal
        firstLine=next(csvreader)[0]
        if firstLine == "first row horizontal":
            firstRowHorizontal=True
        elif firstLine == "first row vertical":
            firstRowHorizontal=False
        else:
            raise Exception("CSV file did not specify whether first row was horizontal or vertical")

        self.updateRawData(csvreader) #store the info in the csv
        self.updateArrowData(firstRowHorizontal)#make the data array and update arrow directions
        self.updateVertexClassifactionDictionary()#make the dictionary of the vertex classification types
        self.updateDimers()#find the dimers

        #find the centers
        if(autoAlignCenters):
            self.updateCenters()
            firstCenter=self.findFirstCenter()
            self.updateCentersFromPattern(firstCenter)
        else:
            if firstCenter is None:
                self.updateCenters()
            else:
                self.updateCentersFromPattern(firstCenter)
        

        self.updateStrings()#find the strings
        #self.removeStringsNotConnectingInteriors()#removestrings that connect less than two interiors
        if(removeEdgeStrings):
            self.removeEdgeStrings();
        #self.removeSmallStrings();
        self.updateCompositeSquareCenters()#find the center of composite squares and count how many strings are in each

        self.imagePaddingConstant=0.05#blank border with as percent of image when we draw it to a cv2 image

        

    #takes a csvreader and updates the rawData array
    def updateRawData(self,csvreader):
        #load the data from the csv into the data array
        self.rawData=[]
        for row in csvreader:
            self.rawData.append([])
            for char in row:
                if(char==" "):
                    self.rawData[len(self.rawData)-1].append(None)
                elif(char==""):
                    pass
                else:
                    self.rawData[len(self.rawData)-1].append(int(char))

    #populates self.data with arrow cells and arrows based on the self.rawData array
    def updateArrowData(self,firstRowHorizontal):
        rawData=self.rawData
        data=[]
        horizontal=firstRowHorizontal#whether the arrows in this row pointi horizontal or vertical
        for rawRow in rawData:
            row=[]
            for rawCell in rawRow:
                if(rawCell==1 or rawCell==-1):
                    if(horizontal):
                        if(rawCell==1):
                            row.append(Cell("left"))
                        else:
                            row.append(Cell("right"))
                    else:
                        if(rawCell==1):
                            row.append(Cell("up"))
                        else:
                            row.append(Cell("down"))
                elif(rawCell==0):
                    cell=Cell(None)
                    cell.badData=True
                    row.append(cell)
                else:
                    row.append(Cell(None))

            data.append(row)
            horizontal=not horizontal
        self.data=data

    #simply stores the vertex counts and types into self.vertexClasses
    def updateVertexClassifactionDictionary(self):

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


        #format: verticies[number of islands][type][arrangementIndex]
        #calculate all possible rotations and inversions of the base verticies and stores that in the verticies dictionary
        verticies={}
        for islandCount in [2,3,4]:#loop through 2, 3, and 4 island vertices
            verticies[islandCount]={}
            for type in baseVerticies[islandCount].keys():#loop through each type in that island count
                verticies[islandCount][type]=[]
                for combination in baseVerticies[islandCount][type]:#now loop through each combination for a give island count and vertex type
                    for i in range(4):#4 islands means four cyclic rotation combinations of a pattern
                        combination=cyclicRotate(combination)#apply teh cyclic rotation

                        #add the rotation and its inverse to the array (if not in their already)
                        if ( combination not in verticies[islandCount][type]):
                            verticies[islandCount][type].append(combination)
                        if( invert(combination) not in verticies[islandCount][type]):
                            verticies[islandCount][type].append(invert(combination))
        self.vertexClasses=verticies
    
    #gives a unique number to each row,col in the grid
    def getRowColIndex(self,row,col):
        return row*len(self.data)+col
    
    #given a string, find what interior centers its touching
    def getTouchingCenters(self, string):
        touchingCenters=[]
        for point in string.getPoints():
            if self.getCell(*point).interiorCenter:
                touchingCenters.append(point)
        
        return touchingCenters
    
    def getTouchingCompositeSquares(self, string):
        touchingCenters=self.getTouchingCenters(string)

        squares=[]
        for center in touchingCenters:
            square=self.getCorrespondingCompositeSquare(*center)
            if square not in squares:
                squares.append(square)
        
        return squares

    
    def getCorrespondingCompositeSquare(self,row,col):
        assert self.getCell(row, col).interiorCenter

        pattern=[(0,1),(0,-1),(1,0),(-1,0)]

        for rowOff,colOff in pattern:
            cell=self.getCell(row+rowOff,col+colOff)
            if(cell and cell.compositeSquareCenter):
                return (row+rowOff,col+colOff)
        
        return None
        


    def getStringID(self,string):#generates an ID for a string based off the rows and columns its touching
        points=string.getPoints()
        
        points=frozenset(points)#sort in case things are in a different order between strings

        return hash(points)%(2**32-1)

    def getStringsByTouchingInteriors(self):
        raise Exception("should not be using this function")
        out={}
        badIDs=[]#bad IDs keeps track of strings with the same ID (we don't want those)
        for string in self.strings:
            if(string.id not in badIDs):#if id is good
                if(string.id in out.keys()):#if a string already has this id, remove both strings and mark id as bad
                    del out[string.id]
                    badIDs.append(string.id)
                else:
                    out[string.id]=string
        return out


    #gets the x spacing and y spacing for an image based on the number of rows, column and image padding
    def getSpacingXSpacingY(self,image):
        imageHeight,imageWidth,imageChannels=image.shape

        #formula, subtract off 2 times the padding, and then divide by the number of rows or cols
        spacingX=(imageWidth-2*self.imagePaddingConstant*imageWidth)/(self.colCount-1)
        spacingY=(imageHeight-2*self.imagePaddingConstant*imageHeight)/(self.rowCount-1)

        return spacingX,spacingY

    #get where a given (row,col) coordinant will appear on an image
    def getXYFromRowCol(self,row,col,image):
        imageHeight,imageWidth,imageChannels=image.shape

        paddingY=self.imagePaddingConstant*imageHeight
        paddingX=self.imagePaddingConstant*imageWidth

        y=paddingY+row/self.rowCount*(imageHeight-2*paddingY)
        x=paddingX+col/self.colCount*(imageWidth-2*paddingX)
        return (int(x),int(y))

    #return true if the cell is at the edge of a group of arrows (edge of the sample or inside of an unknown region)
    def isEdge(self,row,col):
        if(row==0 or col==0 or row==len(self.data)-1 or col==len(self.data[row])-1):
            return True
        if(self.getCell(row,col).center and (row==1 or col==1 or row==len(self.data)-2 or col==len(self.data[row])-2)):
            return True
        return False

    def getVertexEnergy(self,row,col):
        parEnergy=0;
        perpEnergy=0;

        top,right,bottom,left = self.getInOutPattern(row,col)
        
        for pair in [(top,bottom),(right,left)]:
            if pair[0] is None or pair[1] is None:
                continue
            if(pair[0]==pair[1]):
                parEnergy+=1
            else:
                parEnergy-=1
        
        for pair in [(top,right),(right,bottom),(bottom,left),(left,top)]:
            if(pair[0] is None or pair[1] is None):
                continue
            if(pair[0]==pair[1]):
                perpEnergy+=1
            else:
                perpEnergy-=1
        
        return VertexEnergy(parEnergy,perpEnergy)
    
    def getStringEnergy(self,string):
        energy=VertexEnergy(0,0)
        for point in string.getPoints():
            energy+=self.getVertexEnergy(*point)
        return energy



    #properties to return number of rows and columns in grid
    @property
    def rowCount(self):
        return len(self.data)
    @property
    def colCount(self):
        return len(self.data[0])

    #just loop through the strings and draw them
    def drawStrings(self, image,lineWidth=1,showID=False):
        for string in self.strings:
            self.drawString(string,image,lineWidth,showID=showID)
    #draw a given string
    def drawString(self,string,image,lineWidth=1,showID=False):
        #loop through each line segment and draw it
        for line in string.lineSegments:
            start=self.getXYFromRowCol(*line.start,image)
            end=self.getXYFromRowCol(*line.end,image)
            image=cv2.line(image,start,end,string.color,lineWidth)

        """if(len(string.getPoints())>50):
            slope=getSlope(string.getPoints())
            intercept=getIntercept(string.getPoints())
            cv2.line(image,self.getXYFromRowCol(0,intercept,image),self.getXYFromRowCol(1000,slope*1000,image),RED,2)"""

        #cv2.line(image, self.getXYFromRowCol(*string.getCoM(),image), self.getXYFromRowCol(*self.getNearestStringNeighbor(string).getCoM(),image),RED,2)
        if(showID):
            text=str(string.id)
            #text+=" "+str(self.getStringEnergy(string))
            #if(string.isLoop()):
            #    text+=" loop"
            #text+=" "+str(self.getTouchingCompositeSquares(string))
            cv2.putText(image, text, self.getXYFromRowCol(*string.getCoM(),image), cv2.FONT_HERSHEY_SIMPLEX,0.4, BLACK, 1, cv2.LINE_AA)
            

    #draw the line segment representation of each string
    def drawStringTraces(self,image):
        for string in self.strings:
            trace=string.getTrace()
            start=self.getXYFromRowCol(*trace.start,image)
            end=self.getXYFromRowCol(*trace.end,image)
            cv2.line(image,start,end,string.color,3)

            #cv2.putText(image, str(piDist(math.pi/2,trace.angle))[0:5], start, cv2.FONT_HERSHEY_SIMPLEX,0.3, BLACK, 1, cv2.LINE_AA)


    def getTouchingInteriors(self,string):
        touchedInteriors=[]#keep track of the ones we have already touchesd
        for line in string.lineSegments:#loop through each line segment
            start=self.getCell(line.start[0],line.start[1])
            end=self.getCell(line.end[0],line.end[1])

            #if the start or end is touching an interior (that the string has not touched before)
            if start is not None and not start in touchedInteriors and start.interiorCenter:
                touchedInteriors.append(line.start)
            if end is not None and not end in touchedInteriors and end.interiorCenter:
                touchedInteriors.append(line.end)
        
        return touchedInteriors
    #count how many interior centers a given string
    def countTouchingInteriors(self,string):
    
        return len(self.getTouchingInteriors(string))

    def countTouchingEdges(self,string):
        count=0
        touchedInteriors=[]#keep track of the ones we have already touchesd
        for line in string.lineSegments:#loop through each line segment
            if(self.isEdge(line.start[0],line.start[1])):
                count+=1;
            if(self.isEdge(line.end[0],line.end[1])):
                count+=1;

        return count

    #simply removes all strings that touch less than 2 interiors
    def removeStringsNotConnectingInteriors(self):
        self.strings=[string for string in self.strings if self.countTouchingInteriors(string)>=2]


    def removeEdgeStrings(self):
        self.strings=[string for string in self.strings if self.countTouchingInteriors(string)>=2 or self.countTouchingEdges(string)==0]
    def removeStringsTouchingNoCenters(self):
        self.strings=[string for string in self.strings if self.countTouchingInteriors(string)>=1]
    def removeSmallStrings(self):
        self.strings=[string for string in self.strings if string.getLength()>2]


    #just draws all cells in self.data
    def drawCells(self,image,flagCell = lambda row, col : False):
        data=self.data
        #make the line mesh
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                self.drawCell(rowI,colI,image, flagCell=flagCell)

    #draws a given cell
    def drawCell(self,rowI,colI,image, flagCell = lambda row, col : False):
        cell=self.getCell(rowI,colI)
        x, y = self.getXYFromRowCol(rowI,colI,image)
        spacingX,spacingY=self.getSpacingXSpacingY(image)

        #if it is an arrow cell
        if(cell.arrow is not None):
            right=(int(x+spacingX/4),y)
            left=(int(x-spacingX/4),y)

            down=(x,int(y+spacingY/4))
            up=(x,int(y-spacingY/4))

            color=BLACK
            lineWidth=1
            if(flagCell(rowI,colI)):
                color=RED
                lineWidth=3

            tipLength=0.5

            if(cell.arrow=="up"):
                image=cv2.arrowedLine(image,down,up,color,lineWidth,tipLength=tipLength)
            elif(cell.arrow=="down"):
                image=cv2.arrowedLine(image,up,down,color,lineWidth,tipLength=tipLength)
            elif(cell.arrow=="left"):
                image=cv2.arrowedLine(image,right,left,color,lineWidth,tipLength=tipLength)
            elif(cell.arrow=="right"):
                image=cv2.arrowedLine(image,left,right,color,lineWidth,tipLength=tipLength)
        if(cell.dimer):
            cv2.circle(image,(x,y),2,RED,-1)
        if(cell.interiorCenter):
            cv2.circle(image,(x,y),2,BLUE,-1)
        if(cell.compositeSquareCenter):
            cv2.circle(image,(x,y),2,GREEN,-1)
        if(cell.center and not cell.interiorCenter):
            #cv2.circle(image,(x,y),2,BLACK,-1)
            pass
            #cv2.rectangle(image,(int(x-4*spacingX),int(y-4*spacingY)),(int(x+4*spacingX),int(y+4*spacingY)),GREEN,1)

            #cv2.putText(image, str(cell.stringsInCompositeSquare), (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.3, BLACK, 1, cv2.LINE_AA)
        energy=self.getVertexEnergy(rowI,colI)
        #if energy.par != 0 or energy.perp!=0:
        #    cv2.putText(image, str(energy), (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.2, BLACK, 1, cv2.LINE_AA)

    #gets a cell at a given rowI, colI and returns None if that coordinant is out of bounds of the array
    def getCell(self,rowI,colI):
        if(rowI>=0 and rowI<len(self.data) and colI>=0 and colI<len(self.data[rowI])):
            return self.data[rowI][colI]
        else:
            return None

    #gets the four cells next to a given cell (returns None if that cell is out of bounds)
    def getNeighbors(self,rowI,colI):

        top=self.getCell(rowI-1,colI)
        bottom=self.getCell(rowI+1,colI)

        left=self.getCell(rowI,colI-1)
        right=self.getCell(rowI,colI+1)

        return top,right,bottom,left

    #count how many arrows point towards or away a given cell
    def countInOutArrows(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        count=0
        #we cant just simply check if it is an arrow because sometimes a cell has an arrow
        #next to it which does not point at that cell
        if(top is not None and (top.arrow=="up" or top.arrow=="down")):
            count+=1
        if(bottom is not None and (bottom.arrow=="up" or bottom.arrow=="down")):
            count+=1
        if(right is not None and (right.arrow=="right" or right.arrow=="left")):
            count+=1
        if(left is not None and (left.arrow=="right" or left.arrow=="left")):
            count+=1
        return count

    #simply counts the number of neighbors which are arrows
    #(does not consider wheree the arrows are pointing)
    def countArrowNeighbors(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        all=[top,right,bottom,left]
        count=0
        for island in all:
            if island is not None and island.hasArrow:
                count+=1
        return count
    def countBadDataNeighbors(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        all=[top,right,bottom,left]
        count=0
        for island in all:
            if island is not None and island.badData:
                count+=1
        return count



    #returns true if a given cell at (rowI,colI) is the center of a plaquette
    def isCenter(self,rowI,colI):
        cell=self.getCell(rowI,colI)
        if(cell and cell.badData):
            return False
        if(cell is None or cell.hasArrow==True or self.countBadDataNeighbors(rowI,colI)!=0 or self.countArrowNeighbors(rowI,colI)!=0):
            return False


        valid=True

        pattern=[
        [-1,1,-1],
        [1,0,1],
        [0,0,0],
        [1,0,1],
        [-1,1,-1]
        ]
        for checkRowI, patternRowI in zip(range(rowI-2,rowI+3),range(0,5)):
            for(checkColI, patternColI) in zip(range(colI-1,colI+2),range(0,3)):
                if(pattern[patternRowI][patternColI]==-1):
                    continue
                cell=self.getCell(checkRowI,checkColI)
                if(cell is None):
                    valid=False
                else:
                    if(pattern[patternRowI][patternColI]==1 and not cell.hasArrow):
                        valid=False
                    if(pattern[patternRowI][patternColI]==0 and cell.hasArrow):
                        valid=False

        if(valid==True):
            return True
        valid=True

        pattern=[
        [-1,1,0,1,-1],
        [1,0,0,0,1],
        [-1,1,0,1,-1]
        ]
        for checkRowI, patternRowI in zip(range(rowI-1,rowI+2),range(0,3)):
            for(checkColI, patternColI) in zip(range(colI-2,colI+3),range(0,5)):
                if(pattern[patternRowI][patternColI]==-1):
                    continue
                cell=self.getCell(checkRowI,checkColI)
                if(cell is None):
                    valid=False
                else:
                    if(pattern[patternRowI][patternColI]==1 and not cell.hasArrow):
                        valid=False
                    if(pattern[patternRowI][patternColI]==0 and cell.hasArrow):
                        valid=False

        if(valid==False):
            return False

        #if(self.countArrowNeighbors(rowI,colI)==0 and self.countBadDataNeighbors(rowI,colI)==0 and cell is not None and cell.hasArrow==False):
            #return True
        return True

    #returns true if the given cell at (rowI,colI) is an interior center
    def isInteriorCenter(self,rowI,colI):
        if(not self.getCell(rowI,colI).center):
            raise Exception("This function should only be called on centers")


        #the basic algorithm is that if there are two centers seperated vertically or horizontally
        #by a distance of 2, those must be interior centers

        #these are the places to check that would indicate that it is an interior center
        potentialCenters=[(rowI+2,colI),(rowI-2,colI),(rowI,colI+2),(rowI,colI-2)]

        for cell in potentialCenters:
            if(self.isCenter(cell[0],cell[1])):
                return True

        return False

    #returns an array of 4 elements where each element is -1,1, or None representing the 4 neighbors
    #-1 means arrow pointing away, 1 means arrow pointing towarsd, None means no arrow there or arrow not pointing toward/away
    def getInOutPattern(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        pattern=[]
        if top is not None:
            if(top.arrow=="down"):
                pattern.append(1)
            elif(top.arrow=="up"):
                pattern.append(-1)
            elif(top.badData):
                pattern.append(0)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if right is not None:
            if(right.arrow=="left"):
                pattern.append(1)
            elif(right.arrow=="right"):
                pattern.append(-1)
            elif(right.badData):
                pattern.append(0)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if bottom is not None:
            if(bottom.arrow=="up"):
                pattern.append(1)
            elif(bottom.arrow=="down"):
                pattern.append(-1)
            elif(bottom.badData):
                pattern.append(0)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if left is not None:
            if(left.arrow=="right"):
                pattern.append(1)
            elif(left.arrow=="left"):
                pattern.append(-1)
            elif(left.badData):
                pattern.append(0)
            else:
                pattern.append(None)
        else:
            pattern.append(None)

        return pattern

    #check if a cell at a given row,col is a dimer
    def isDimer(self, rowI,colI):
        cell=self.getCell(rowI,colI)

        vertexPattern=self.getInOutPattern(rowI,colI)
        islandCount=self.countInOutArrows(rowI,colI)

        vertexType=0;

        if islandCount>1 and cell.arrow is None:
            for type in self.vertexClasses[islandCount].keys():#check if the vertex pattern is in the given type
                if vertexPattern in self.vertexClasses[islandCount][type]:
                    vertexType=type
                    break

        return vertexType>1#all non-type-1 verticies are dimers

    #just find all dimers and mark that in their cell object
    def updateDimers(self):
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                cell.dimer=self.isDimer(rowI,colI)
    
    def countArrows(self):
        total=0
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                if(cell.hasArrow):
                    total+=1
        return total
    
    def isBlank(self):
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                if(cell.hasArrow):
                    return False
        return True

    #just find all centers and update that in the cell object
    def updateCenters(self):
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                cell.center=self.isCenter(rowI,colI)
                if(cell.center):
                    cell.interiorCenter=self.isInteriorCenter(rowI,colI)
    
    def findFirstCenter(self):
        centerRowCount=[0]*16
        centerColCount=[0]*16
        for (rowI, row) in enumerate(self.data):
            for (colI, cell) in enumerate(row):
                if(cell.center and self.isInteriorCenter(rowI, colI)):
                    if(self.getCell(rowI+2,colI).center and  self.isInteriorCenter(rowI+2, colI)):

                        rowOff=(colI%16>=8)
                        colOff=((rowI+1)%16>=8) 



                        centerRowCount[((rowI+1) +rowOff*8 )%16]+=1
                        centerColCount[((colI) +colOff*8 )%8  ]+=1
                    elif(self.getCell(rowI,colI+2).center and self.isInteriorCenter(rowI, colI+2)):
                        pass
                        #centerRowCount[((rowI) +((colI+1)%16>=8)*8 )%16]+=1
                        #centerColCount[((colI+1) +((rowI)%16>=8)*8 )%16  ]+=1
        
        rowChoice=-1
        rowMaxCount=-1
        colChoice=-1
        colMaxCount=-1


        for (i,count) in enumerate(centerRowCount):
            if(count>rowMaxCount):
                rowChoice=i
                rowMaxCount=count
        
        for (i,count) in enumerate(centerColCount):
            if(count>colMaxCount):
                colChoice=i
                colMaxCount=count
        

        return (rowChoice-16, colChoice-16)

    def updateCentersFromPattern(self,firstCenter):

        #first clear all centers
        for row in self.data:
            for cell in row:
                cell.center=False
                cell.interiorCenter=False

        initialCoord=np.asarray(firstCenter)
        
        upDownCenters=[(0,-3),(0,3),(-3,2),(-3,-2),(3,2),(3,-2)]
        leftRightCenters=[(-3,0),(3,0),(2,3),(2,-3),(-2,3),(-2,-3)]

        rowStartUpDown=True
        
        row=initialCoord[0]
        while(row<len(self.data)+8):
            col=initialCoord[1]
            upDown=rowStartUpDown
            while(col<len(self.data[0])+8):

                #update Interior centers
                if(upDown):
                    cell1=self.getCell(row-1,col)
                    cell2=self.getCell(row+1,col)
                else:
                    cell1=self.getCell(row,col-1)
                    cell2=self.getCell(row,col+1)
                if(cell1 is not None):
                    cell1.interiorCenter=True
                    cell1.center=True
                if(cell2 is not None):
                    cell2.interiorCenter=True
                    cell2.center=True

                if(upDown):
                    centerCoords=upDownCenters
                else:
                    centerCoords=leftRightCenters
                for coord in centerCoords:
                    cell=self.getCell(row+coord[0],col+coord[1])
                    if cell is not None:
                        cell.center=True

                col+=8
                upDown=not upDown
            rowStartUpDown=not rowStartUpDown
            row+=8

    #check if the given rowI,colI is the center of a composite square
    def isCompositeSquareCenter(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        if(top is not None and top.interiorCenter and bottom is not None and bottom.interiorCenter):
            return True
        if(right is not None and right.interiorCenter and left is not None and left.interiorCenter):
            return True

    #count the number of strings within a distance of a given rowI,colI
    #this does not use distance like a circle, instead it is the "radius" of a square
    #e.g. squareRadiu=2 will look in a 5x5 square centered around the row,col
    def getStringsInSquare(self,rowI,colI,squareRadius):
        minRow=rowI-squareRadius
        minCol=colI-squareRadius
        maxRow=rowI+squareRadius
        maxCol=colI+squareRadius

        count=0;

        for string in self.strings:
            for line in string.lineSegments:
                start=line.start
                end=line.end
                if start[0]>=minRow and start[0]<=maxRow and start[1]>=minCol and start[1]<=maxCol:
                    count+=1
                    break
                if end[0]>=minRow and end[0]<=maxRow and end[1]>=minCol and end[1]<=maxCol:
                    count+=1
                    break
        return count

    #mark all composite square centers in their cell class
    def updateCompositeSquareCenters(self):
        for (rowI, row) in enumerate(self.data):
            for(colI, cell) in enumerate(row):
                cell.compositeSquareCenter=self.isCompositeSquareCenter(rowI,colI)
                if(cell.compositeSquareCenter):
                    cell.stringsInCompositeSquare=self.getStringsInSquare(rowI,colI,4)

    #gets the centers  of placquettes which a given Row,col is on the edge of
    def getNearbyCenterCoords(self,rowI,colI):
        centers=[]
        for rowOff in range(-2,3):
            for colOff in range(-2,3):
                cell=self.getCell(rowI+rowOff,colI+colOff)
                if( cell is not None and cell.center ):
                    centers.append((rowI+rowOff,colI+colOff))
        return centers

    #given a row column, returns an array of the coordinates of centers it connects to
    def getStringConnections(self,rowI,colI):

        if not self.getCell(rowI,colI).dimer:
            return []

        islandCount=self.countArrowNeighbors(rowI,colI)
        top,right,bottom,left = self.getNeighbors(rowI,colI)

        #get the 3 or four centers it could connect to
        centers=self.getNearbyCenterCoords(rowI,colI)

        connections=[]

        for i in range(len(centers)):
            center=centers[i]
            relativeRow=center[0]-rowI
            relativeCol=center[1]-colI

            #these are basically the hardcoded rules for how a string can connect through a vertex
            if(relativeRow>0 and relativeCol>0): #bottom right
                if(bottom.arrow=="up" and  right.arrow=="left" or bottom.arrow=="down" and right.arrow=="right"):
                    connections.append(center)
            elif(relativeRow<0 and relativeCol>0): #top right
                if(top.arrow=="up" and  right.arrow=="right" or top.arrow=="down" and right.arrow=="left"):
                    connections.append(center)
            elif(relativeRow<0 and relativeCol<0):# top left
                if(top.arrow=="up" and  left.arrow=="left" or top.arrow=="down" and left.arrow=="right"):
                    connections.append(center)
            elif(relativeRow>0 and relativeCol<0):#bottom left
                if(bottom.arrow=="down" and  left.arrow=="left" or bottom.arrow=="up" and left.arrow=="right"):
                    connections.append(center)
            elif(relativeRow==0 and top is not None and bottom is not None):#right or left
                if(islandCount==2 and (top.arrow != bottom.arrow) or islandCount==3 and top.arrow==bottom.arrow):
                    connections.append(center)
            elif(relativeCol==0 and right is not None and left is not None):#top or bottom
                if(islandCount==2 and right.arrow!=left.arrow or islandCount==3 and right.arrow==left.arrow):
                    connections.append(center)
        return connections

    #update the strings array based on a given cell
    def updateString(self,rowI,cellI):
        #if it is not a dimer, it won't impact the strings
        if not self.getCell(rowI,cellI).dimer:
            return

        #coordinants of the centers it connects to
        connections=self.getStringConnections(rowI,cellI)

        #make lines between the cell and its connections
        lines=[]
        for c in connections:
            lines.append(LineSegment(rowI,cellI,c[0],c[1]))

        for line in lines: #loop through all the line segments in that vertex

            stringIndices=[]
            for (stringI,string) in enumerate(self.strings):#loop through all strings
                for stringLine in string.lineSegments:#if it connects with that string
                    if line.connectsWith(stringLine) and not(self.getCell(*line.getSharedPoint(stringLine)).interiorCenter)and stringI not in stringIndices:
                        stringIndices.append(stringI)

            if(len(stringIndices)==0):#if it is not touching any strings, make a new string group
                self.strings.append(String([line]))
            elif(len(stringIndices)==1):#if it is touching one string, add it to that one
                self.strings[stringIndices[0]].addLineSegment(line)
            else: #touching multiple strings which need to be combined
                newString=String([])#combination of toching line groups
                for stringI in stringIndices:
                    newString.addString(self.strings[stringI])#add group to new group
                    self.strings[stringI]=None#set it to None instead of delete so theres no indexing issues
                self.strings=[string for string in self.strings if string is not None]#now delete the old string
                newString.addLineSegment(line)#add the line to the new, larger string
                self.strings.append(newString)#add the larger group back into the list of line groups

    #just call the updateSTring function on all cells
    def updateStrings(self):
        self.strings=[]
        for (rowI, row) in enumerate(self.data):
            for(cellI, cell) in enumerate(row):
                self.updateString(rowI,cellI)
        
        for string in self.strings:
            id=self.getStringID(string)
            if not self.randomizeStringColor:
                np.random.seed(id)
            string.color=list(np.random.random(size=3) * 256)
            string.id=id

            string.touchedInteriors=self.getTouchingInteriors(string)
    #returns a list of the angles of the strings
    def getStringAngles(self):
        angles=[]
        for string in self.strings:
            angles.append(string.getTrace().angle)
        return angles
    #returns the number of composite squares
    def numCompositeSquares(self):
        count=0
        for row in self.data:
            for cell in row:
                if cell.compositeSquareCenter:
                    count+=1
        return count

    #takes a string and returns a string which is its nearest neighbor
    def getNearestStringNeighbor(self,string):
        CoM=string.getCoM()
        minDist=10000000000;
        nearest=None
        for string2 in self.strings:
            if string2!=string:
                distance=dist(CoM,string2.getCoM())
                if(distance<minDist):
                    minDist=distance
                    nearest=string2
        return nearest

    #uses the algorithm to find the correlation of teh overall sample
    def getStringCorrelation(self):
        correlation=0;
        count=0;
        for string1 in self.strings:
            string1Trace=string1.getTrace();
            for string2 in self.strings:
                if(string1 != string2):
                    string2Trace=string2.getTrace();
                    correlation+=string1Trace.getCorrelation(string2Trace);
                    count+=1;
        if(count==0):
            return 0
        return correlation/count

    def getNearestNeighborStringCorrelation(self):
        correlation=0;
        count=0;
        for string1 in self.strings:
            string2=self.getNearestStringNeighbor(string1)
            if(string2 is None):#there is only one string
                return 0
            correlation+=string1.getTrace().getCorrelation(string2.getTrace())
            count+=1;
        if(count==0):
            return 0
        return correlation/count;


        """totalDotProduct=0
        nearestDotProduct=0
        for string1 in self.strings:
            string1Trace=string1.getTrace()
            neighbor=self.getNearestStringNeighbor(string1)
            if(neighbor==None):#there is only one string on the board
                return 0
            neighborTrace=neighbor.getTrace()
            nearestDotProduct+=string1Trace.getCorrelation(neighborTrace)

            for string2 in self.strings:
                if string1 != string2:
                    string2Trace=string2.getTrace()
                    totalDotProduct+=string1Trace.getCorrelation(string2Trace)


        return nearestDotProduct/(totalDotProduct-nearestDotProduct)"""




    """def getCompositeSquareStringCounts(self):
        counts=[]
        totalCompositeSquares=self.numCompositeSquares()
        for row in self.data:
            rowCount=[]
            for cell in row:
                if cell.compositeSquareCenter:
                    rowCount.append(cell.stringsInCompositeSquare/totalCompositeSquares)
            if rowCount !=[]:
                counts.append(rowCount)
        return counts"""



    """def getStringCorrelationVsDistance(self):
        dataPoints=[]
        for stringi in range(len(self.strings)):
            for stringj in range(stringi):
                string1=self.strings[stringi]
                string2=self.strings[stringj]

                distance=dist(string1.getCoM(),string2.getCoM())
                angleDif=piDist(string1.getTrace().angle, string2.getTrace().angle)
                sizeDif=string1.getTrace().length-string2.getTrace().length

                dataPoints.append([distance,angleDif,sizeDif])
        return dataPoints"""


if __name__=="__main__":
    if(False):
        parser = argparse.ArgumentParser(description='Santa fe csv analysis')
        parser.add_argument("file", type=str, help="Path to csv file")
        args=parser.parse_args()


        try:
            file=open(args.file,newline="\n")
        except:
            raise Exception("Error with file")
        lattice=SantaFeLattice(file)

        outString=args.file+", "
        lattice.removeStringsNotConnectingInteriors()
        outString+=""+str(len(lattice.strings)/lattice.numCompositeSquares())+", "
        lattice.removeSmallStrings()
        outString+=""+str(lattice.getStringCorrelation())+", "
        outString+=""+str(lattice.getNearestNeighborStringCorrelation())+"\n"

        with open("out.txt","a") as file:
            file.write(outString)


        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        lattice.drawCells(outputImage)
        lattice.drawStrings(outputImage)
        lattice.drawStringTraces(outputImage)


        cv2.imshow("window",outputImage)
        cv2.waitKey(0)

        outputFileName=args.file.split(".")[0]+"_output.jpg"
        cv2.imwrite(outputFileName, np.float32(outputImage));
    
    else:
        todo=['square/square674.csv',
        'square/square675.csv',
        'square/square676.csv',
        'square/square677.csv',
        'square/square678.csv',
        'square/square679.csv',
        'square/square680.csv',
        'square/square681.csv',
        'santafeanalysis/CaptureFile0002.csv',
        'santafeanalysis/CaptureFile0003.csv',
        'santafeanalysis/CaptureFile0004.csv',
        'santafeanalysis/CaptureFile0005.csv',
        'santafeanalysis/CaptureFile0006.csv',
        'santafeanalysis/CaptureFile0007.csv',
        'santafeanalysis/CaptureFile0008.csv',
        'santafeanalysis/CaptureFile0009.csv',
        'santafeanalysis/CaptureFile0010.csv',
        'santafeanalysis/CaptureFile0011.csv',
        'santafeanalysis/CaptureFile0012.csv',
        'santafeanalysis/CaptureFile0013.csv',
        'santafeanalysis/CaptureFile0033.csv',
        'santafeanalysis/CaptureFile0034.csv',
        'santafeanalysis/CaptureFile0035.csv',
        'santafeanalysis/CaptureFile0036.csv',
        'santafeanalysis/CaptureFile0048.csv',
        'santafeanalysis/CaptureFile0049.csv',
        'santafeanalysis/CaptureFile0050.csv',
        'santafeanalysis/CaptureFile0051.csv',
        'santafeanalysis/CaptureFile0052.csv',
        'santafeanalysis/CaptureFile0053.csv',
        ]
        for fileName in todo:
            try:
                file=open(fileName,newline="\n")
            except:
                raise Exception("Error with file")
            lattice=SantaFeLattice(file)

            outString=fileName+", "
            #lattice.removeStringsNotConnectingInteriors()
            outString+=""+str(len(lattice.strings)/lattice.numCompositeSquares())+", "
            lattice.removeSmallStrings()
            outString+=""+str(lattice.getStringCorrelation())+", "
            outString+=""+str(lattice.getNearestNeighborStringCorrelation())+"\n"

            with open("out.txt","a") as file:
                file.write(outString)


            outputImage=np.zeros((1000,1000,3), np.uint8)
            outputImage[:,:]=(250,250,250)
            lattice.drawCells(outputImage)
            lattice.drawStrings(outputImage)
            lattice.drawStringTraces(outputImage)


            #cv2.imshow("window",outputImage)
            #cv2.waitKey(0)
            print(fileName)

            outputFileName=fileName.split(".")[0]+"_output.jpg"
            cv2.imwrite(outputFileName, np.float32(outputImage));

