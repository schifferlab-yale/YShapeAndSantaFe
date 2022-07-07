import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
import csv
import matplotlib as plt
import os

from numpy.lib.function_base import average

#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)
ORANGE=(0,165,255)
PURPLE=(130,0,75)

#Each of the four points in an island
TOPLEFT=0
TOPRIGHT=1
MIDDLE=2
BOTTOM=3


#for indexing the neighbors of an island
TOPLEFTNEIGHBOR=0
TOPRIGHTNEIGHBOR=1
LEFTNEIGHBOR=2
RIGHTNEIGHBOR=3
BOTTOMLEFTNEIGHBOR=4
BOTTOMRIGHTNEIGHBOR=5

#"Root 3 over 2"
R3O2=math.sqrt(3)/2

def getFileData(file):
    file=file.read().replace("\t","")
    
    file=file.replace("\r","")#some systems use \r\n, others use \n
    file=file.split("\n")
    
    file=[line.split(", ") for line in file]
    return file





#basic class for the chargegrid
class Charge:
    def __init__(self,x,y,charge,type=None):
        self.x=x
        self.y=y
        self.charge=charge
        self.type=type
    def __repr__(self):
        #return f"{self.charge}"
        return f"{self.charge} at {self.x},{self.y}"

#basic moment class for MomentGrid
class Moment:
    def __init__(self,x,y,angle,magnitude):
        if angle is None: raise Exception("Angle is None")
        self.x=x
        self.y=y
        self.angle=angle
        self.magnitude=magnitude

    def __repr__(self):
        return f"{restrictAngleRad(self.angle)*180/math.pi}"
        return f"{restrictAngleRad(self.angle)*180/math.pi} {self.x} {self.y}"

def angleDotProductRad(angle1,angle2):
    return math.cos(angle1-angle2)

#island has any 0s in it (bad data)
def isBadIsland(island):
    for arm in island:
        if arm==0:
            return True
    return False

#gets a numpy array [x,y] for the moment direction of an island
def getIslandMomentVector(island):
    if isBadIsland(island):
        return None
    topLeft=np.array([-R3O2,-0.5])*-island[TOPLEFT]
    topRight=np.array([R3O2,-0.5])*-island[TOPRIGHT]
    bottom=np.array([0,1])*-island[BOTTOM]
    return topLeft+topRight+bottom

#gets angle of island moment vector (in degrees)
def getIslandAngle(island):
    if island is None:
        return None
    vector=getIslandMomentVector(island)
    if vector[0]==0 and vector[1]==0:
        return None
    angleRad=math.atan2(vector[1],vector[0])
    angle=int(angleRad/(2*math.pi)*360)
    return angle

#restricts a number to be in between [0,2pi)
def restrictAngleRad(angle):
    while angle<0:
        angle+=2*math.pi;
    while angle>=2*math.pi:
        angle-=2*math.pi
    return angle

#restricts a number to be in between [0,360)
def restrictAngleDeg(angle):
    if angle is None: return None

    while angle<0:
        angle+=360;
    while angle>=360:
        angle-=360
    return angle

#converts an island to an angle
def getIslandAngleRad(island):
    if isBadIsland(island):
        return None
    vector=getIslandMomentVector(island)
    if vector[0]==0 and vector[1]==0:
        return None
    angleRad=math.atan2(vector[1],vector[0])
    return angleRad


#https://stackoverflow.com/questions/57400584/how-to-map-a-range-of-numbers-to-rgb-in-python
def num_to_rgb(val, max_val=1):
    i = (val * 255 / max_val);
    r = round(math.sin(0.024 * i + 0) * 127 + 128);
    g = round(math.sin(0.024 * i + 2) * 127 + 128);
    b = round(math.sin(0.024 * i + 4) * 127 + 128);
    return (r,g,b)


class MomentGrid:
    def __init__(self):
        self.moments=[]
        self.distError=0.00000001
    
    def addMoment(self,moment):
        self.moments.append(moment)
    
    def draw(self,img,padding=50):
        xS=[moment.x for moment in self.moments]
        yS=[moment.y for moment in self.moments]
        minX=np.min(xS)
        maxX=np.max(xS)
        minY=np.min(yS)
        maxY=np.max(yS)

        height,width,channels=img.shape

        imgMinX=padding
        imgMinY=padding
        imgMaxX=width-padding
        imgMaxY=height-padding

        imgMax=min(imgMaxX,imgMaxY)
        imgMin=max(imgMinY,imgMinX)

        for moment in self.moments:
            imgX=np.interp(moment.x,[minX,maxX], [imgMin,imgMax])
            imgY=np.interp(moment.y,[minY,maxY], [imgMin,imgMax])

            spacing=(imgMax-imgMin)/math.sqrt(len(self.moments))

            cv2.circle(img,(int(imgX),int(imgY)),2,  BLACK,-1)

            arrowLength=spacing/4

            """testing"""
            #groups=self.getGroupedMomentsByDistance(moment.x,moment.y)

            #print(nearest)

            #cv2.putText(img,str(len(nearest)),(int(imgX+2),int(imgY-2)),cv2.FONT_HERSHEY_COMPLEX, 0.3, BLACK)

            end=(int(imgX+math.cos(moment.angle)*arrowLength),int(imgY+math.sin(moment.angle)*arrowLength))
            start=(int(imgX-math.cos(moment.angle)*arrowLength),int(imgY-math.sin(moment.angle)*arrowLength))
            cv2.arrowedLine(img,start,end,BLACK,2,tipLength=0.2)
    
    #get all the moments sorted by their distance from (x,y)
    def momentsByDistance(self,x,y):
        out=[]
        for moment in self.moments:
            dist=math.sqrt((moment.x-x)**2+(moment.y-y)**2)
            out.append((dist,moment))

        out=sorted(out,key=lambda el:el[0])
        return out
    
    #group moments by their distance from a given point 
    def getGroupedMomentsByDistance(self,x,y):
        error=self.distError#for floating point error in distances

        momentsByDist=self.momentsByDistance(x,y)
        groupedMoments=[]
        lastDist=-100
        for dist,moment in momentsByDist:
            if abs(dist-lastDist)<error:
                groupedMoments[-1][1].append(moment)
            else:
                groupedMoments.append((dist,[moment]))
                lastDist=dist

        return groupedMoments

    def getNthClosestMoments(self,x,y,n):
        return self.getGroupedMomentsByDistance(x,y)[n][1]

    def getRelativeAngleVsDistance(self,maxNAway):


        angleDiffSum=[0]*maxNAway
        angleDiffCount=[0]*maxNAway

        angleCounts=[]
        for i in range(maxNAway):angleCounts.append({0:0,60:0,120:0,180:0,240:0,300:0})

        for moment in self.moments:
            groups=self.getGroupedMomentsByDistance(moment.x,moment.y)
            for n in range(maxNAway):
                if moment.angle is None:
                    continue

                otherMoments=groups[n][1]
                for otherMoment in otherMoments:
                    if otherMoment.angle is None:
                        continue
                    angle=round(180/math.pi*restrictAngleRad(otherMoment.angle-moment.angle))

                    angleDiffSum[n]+=angle
                    angleDiffCount[n]+=1

                    if angle in angleCounts[n].keys():
                        angleCounts[n][angle]+=1
                    else:
                        raise "Bad angle"
            
        anglePercent=[]
        for counts in angleCounts:
            total=0
            for key in counts.keys():
                total+=counts[key]
            dict={}
            for key in counts.keys():
                dict[key]=counts[key]/total
            anglePercent.append(dict)

        return anglePercent


    def getCorrelationByOffsetAngle(self):

        #format: data[centerMomentAngle][angleToOtherIsland]=correlation

        dataSum={}#sum of the corrolations for a given center moment angle and and offset angle
        dataCount={}#counts the total number of data points (for averaging)

        possibleCenterAngles=[30,90,150,210,270,330]#possible angles an island can be at
        possibleOffsetAngles=[0,60,120,180,240,300]#possible differences between island angles


        #populate empty dicts for all center and offset compbinations
        for i in possibleCenterAngles:
            dataSum[i]={}
            dataCount[i]={}
            for j in possibleOffsetAngles:
                dataSum[i][j]=0
                dataCount[i][j]=0
        
        for moment in self.moments:#loop through all islands in sample
            if moment.angle is None:#skip bad islands
                continue

            nearestNeighbors=self.getGroupedMomentsByDistance(moment.x,moment.y)[1]#returns an array islands at a distance of 1 away from center island
            #we take the first element because we are taking the group of closest islands

            assert nearestNeighbors[0]-1<0.0000001#confirm that the islands are 1 unit away

            centerAngle=round(restrictAngleRad(moment.angle)/math.pi*180)#make sure angle is between [0,2pi) and convert to degrees

            assert centerAngle in possibleCenterAngles

            for neighbor in nearestNeighbors[1]:#take element 1 from nearest neighbors (this is the array of neighbors)
                directionVector=(neighbor.x-moment.x,neighbor.y-moment.y)#vector pointing from the center island to the neighbor
                offsetAngle=round(restrictAngleRad(math.atan2(directionVector[1],directionVector[0]))*180/math.pi,4)#angle between the moment of the two islands
                assert offsetAngle in possibleOffsetAngles

                if moment.angle is not None and neighbor.angle is not None:
                    dataSum[centerAngle][offsetAngle]+=math.cos(moment.angle-neighbor.angle)
                    dataCount[centerAngle][offsetAngle]+=1
        

        dataAvg={}
        for key in dataSum.keys():
            dataAvg[key]={}
            for key2 in dataSum[key].keys():


                if dataCount[key][key2]==0:
                    dataAvg[key][key2]=0
                    #prevent divide by 0
                else:
                    dataAvg[key][key2]=dataSum[key][key2]/dataCount[key][key2]
        return dataAvg


                
    def getCorrelationVsDistance(self,maxNAway):
        correlationCount=[0]*maxNAway
        correlationSum=[0]*maxNAway

        for moment in self.moments:
            if moment.angle is None:
                continue

            groups=self.getGroupedMomentsByDistance(moment.x,moment.y)

            for n in range(maxNAway):
                otherMoments=groups[n][1]
                for otherMoment in otherMoments:

                    if otherMoment.angle is None:
                        continue
                    
                    dotProduct=angleDotProductRad(otherMoment.angle,moment.angle)

                    correlationSum[n]+=dotProduct
                    correlationCount[n]+=1
        
        correlation=[correlationSum[i]/correlationCount[i] for i in range(maxNAway)]
        return correlation

    def getCorrelationVsAbsoluteAngle(self,maxNAway):
        #correlationVsAngle[integer distance][angle]

        correlationSums=[]
        correlationCounts=[]
        for i in range(maxNAway):correlationSums.append({})
        for i in range(maxNAway):correlationCounts.append({})

        for moment in self.moments:
            if moment.angle is None:
                continue

            groups=self.getGroupedMomentsByDistance(moment.x,moment.y)
            for n in range(maxNAway):
                otherMoments=groups[n][1]
                for otherMoment in otherMoments:
                    if otherMoment.angle is None:
                        continue

                    directionVector=(otherMoment.x-moment.x,otherMoment.y-moment.y)
                    relativeAngle=round(restrictAngleRad(math.atan2(directionVector[1],directionVector[0]))*180/math.pi,4)
                    
                    dotProduct=angleDotProductRad(otherMoment.angle,moment.angle)

                    if relativeAngle in correlationCounts[n].keys():
                        correlationCounts[n][relativeAngle]+=1
                        correlationSums[n][relativeAngle]+=dotProduct
                    else:
                        correlationCounts[n][relativeAngle]=1
                        correlationSums[n][relativeAngle]=dotProduct
        

        correlations=[]
        for i in range(len(correlationCounts)):
            correlations.append({})
            for key in correlationCounts[i].keys():
                correlations[i][key]=correlationSums[i][key]/correlationCounts[i][key]
        
        return correlations

    def getDeltaNuCorrelation(self, dist):
        deltaSum=0
        deltaTotal=0
        nuSum=0
        nuTotal=0

        for moment in self.moments:
            groups=self.getGroupedMomentsByDistance(moment.x,moment.y)
            group=[group for group in groups if abs(group[0]-dist)<self.distError][0][1]#all moments at that distance

            for otherMoment in group:

                    directionVector=(otherMoment.x-moment.x,otherMoment.y-moment.y)
                    relativeAngle=round(restrictAngleRad(math.atan2(directionVector[1],directionVector[0]))*180/math.pi,4)
                    correlation=angleDotProductRad(otherMoment.angle,moment.angle)

                    if abs(relativeAngle)<self.distError or abs(relativeAngle-180)<self.distError:
                        deltaTotal+=1
                        deltaSum+=correlation
                    else:
                        nuTotal+=1
                        nuSum+=correlation

        print(deltaTotal)
        print(nuTotal)
        return deltaSum/deltaTotal, nuSum/nuTotal






#stores a list of "Charge" instances and their locations
class ChargeGrid:
    def __init__(self):
        self.charges=[]
        self.distError=0.00000001
    def addCharge(self,charge):
        self.charges.append(charge)
    def draw(self,img,padding=50,colorByType=False):

        xS=[charge.x for charge in self.charges]
        yS=[charge.y for charge in self.charges]
        minX=np.min(xS)
        maxX=np.max(xS)
        minY=np.min(yS)
        maxY=np.max(yS)

        height,width,channels=img.shape

        imgMinX=padding
        imgMinY=padding
        imgMaxX=width-padding
        imgMaxY=height-padding

        imgMax=min(imgMaxX,imgMaxY)
        imgMin=max(imgMinY,imgMinX)

        for charge in self.charges:
            imgX=np.interp(charge.x,[minX,maxX], [imgMin,imgMax])
            imgY=np.interp(charge.y,[minY,maxY], [imgMin,imgMax])

            if colorByType:
                if charge.type=="a":color=RED
                else:color=BLUE
            else:
                if(charge.charge<0):color=BLACK
                elif(charge.charge>0):color=WHITE
                else: color=GREEN

            cv2.circle(img,(int(imgX),int(imgY)),2*abs(charge.charge), color,-1)

    #get all the charges sorted by their distance from (x,y)
    def chargesByDistance(self,x,y):
        out=[]
        for charge in self.charges:
            dist=math.sqrt((charge.x-x)**2+(charge.y-y)**2)
            out.append((dist,charge))

        out=sorted(out,key=lambda el:el[0])
        return out
    
    #group charges by their distance from a given point 
    def getGroupedChargesByDistance(self,x,y):
        error=self.distError#for floating point error in distances

        chargesByDist=self.chargesByDistance(x,y)
        groupedCharges=[]
        lastDist=-100
        for dist,charge in chargesByDist:
            if abs(dist-lastDist)<error:
                groupedCharges[-1][1].append(charge)
            else:
                groupedCharges.append((dist,[charge]))
                lastDist=dist

        return groupedCharges

    def getChargeCorrelation(self,charge):
        groupedCharges=self.getGroupedChargesByDistance(charge.x,charge.y)
        correlations=[]
        for dist,charges in groupedCharges:
            sum=0
            for otherCharge in charges:
                sum+=otherCharge.charge*charge.charge
            avg=sum/len(charges)
            correlations.append((dist,avg))
        
        return correlations
    
    def getAllCorrelationPairs(self,maxDist=100000,shouldCompare=lambda charge1, charge2: True ):
        pairs=[]
        for charge in self.charges:
            for otherCharge in self.charges:
                if shouldCompare(charge,otherCharge):
                    dist=math.sqrt((charge.x-otherCharge.x)**2+(charge.y-otherCharge.y)**2)
                    if dist<=maxDist+self.distError:
                        correlation=charge.charge*otherCharge.charge
                        pairs.append((dist,correlation))
        return pairs
    
    def getOverallChargeCorrelation(self,maxDist=1000000,shouldCompare=lambda charge1, charge2: True ):
        correlations=self.getAllCorrelationPairs(maxDist=maxDist,shouldCompare=shouldCompare)
        correlations=sorted(correlations,key=lambda el:el[0])

        error=self.distError
        groupedCorrelations=[]
        lastDist=-100
        for dist,correlation in correlations:
            if abs(dist-lastDist)<error:
                groupedCorrelations[-1][1].append(correlation)
            else:
                groupedCorrelations.append((dist,[correlation]))
                lastDist=dist

        averageCorrelations=[]
        for el in groupedCorrelations:
            averageCorrelations.append((el[0],np.average(el[1])))
        return averageCorrelations
        



class YShapeLattice:
    def __init__(self,data):
        self.saveData(data)

        
    def saveData(self,inputData):
        #get offset
        if(inputData[0][0]=="first row offset"):
            self.firstRowOffset=True
        elif(inputData[0][0]=="second row offset"):
            self.firstRowOffset=False
        else:
            raise Exception("no offset data")

        #convert file to data array of ints
        data=[]
        for row in range(2,len(inputData)):
            data.append([])
            for col in range(len(inputData[row])):
                if(inputData[row][col] !=""):
                    data[len(data)-1].append(int(inputData[row][col]))

        #group rows into arrays of 4 to represent each island
        newData=[]
        for row in data:
            newData.append([])
            newData[len(newData)-1].append([])
            i=0
            for char in row:
                lastRow=newData[len(newData)-1]
                lastRow[len(lastRow)-1].append(char)
                i+=1
                if(i==4):
                    lastRow.append([])
                    i=0
            lastRow=newData[len(newData)-1]


        #remove empyt islands
        for i,row in enumerate(newData):
            newData[i]=[island for island in row if island!=[]]
        self.data=newData

    def isRowOffset(self,rowI):
        return (rowI%2==0 and self.firstRowOffset) or (rowI%2==1 and not self.firstRowOffset)
    

    def getCoordOrNone(self,row,col):
        if row<0 or col<0 or row>len(self.data)-1 or col>len(self.data[row])-1:
            return None
        else:
            return (row,col)


    #takes a row,col index of a given island and returns the neighbor coords in the form:
    #[topLeft,topRight,left,right,bottomLeft,bottomRight]
    #each element of above array is (row,col)
    def getNeighborsCoords(self,row,col):
        neighbors=[]
        offset=self.isRowOffset(row)

        #top left
        if offset:
            neighbors.append(self.getCoordOrNone(row-1,col))
        else:
            neighbors.append(self.getCoordOrNone(row-1,col-1))
        
        #top right
        if offset:
            neighbors.append(self.getCoordOrNone(row-1,col+1))
        else:
            neighbors.append(self.getCoordOrNone(row-1,col))
        
        #left
        neighbors.append(self.getCoordOrNone(row,col-1))
        #right
        neighbors.append(self.getCoordOrNone(row,col+1))

        #bottom left
        if offset:
            neighbors.append(self.getCoordOrNone(row+1,col))
        else:
            neighbors.append(self.getCoordOrNone(row+1,col-1))

        #bottom right
        if offset:
            neighbors.append(self.getCoordOrNone(row+1,col+1))
        else:
            neighbors.append(self.getCoordOrNone(row+1,col))
        
        return neighbors
    
    def getNeighbors(self,row,col):
        coords=self.getNeighborsCoords(row,col)

        out=[]
        for n in coords:
            if n is None:
                out.append(n)
            else:
                out.append(self.data[n[0]][n[1]])
        return out

    def getVerticies(self,row,col):
        neighbors=self.getNeighbors(row,col)
        thisCell=self.data[row][col]

        if thisCell==[]:
            thisCell=[0,0,0,0]

        for i in range(len(neighbors)):
            if neighbors[i] ==[] or neighbors[i] is None:
                neighbors[i]=[0,0,0,0]

        topLeft=-(neighbors[TOPLEFTNEIGHBOR][BOTTOM]+neighbors[LEFTNEIGHBOR][TOPRIGHT]+thisCell[TOPLEFT])
        topRight=-(neighbors[TOPRIGHTNEIGHBOR][BOTTOM]+neighbors[RIGHTNEIGHBOR][TOPLEFT]+thisCell[TOPRIGHT])
        bottom=-(neighbors[BOTTOMLEFTNEIGHBOR][TOPRIGHT]+neighbors[BOTTOMRIGHTNEIGHBOR][TOPLEFT]+thisCell[BOTTOM])

        return [topLeft,topRight,bottom]
    
    def getIslandCharge(self,row,col):
        island=self.data[row][col]
        return (island[TOPLEFT]+island[TOPRIGHT]+island[BOTTOM])

    def draw(self,img,showIslands=True, showIslandCharge=False, twoColor=False, showRings=False,showVertexCharge=False, halfInverted=False, simplifyIslands=False, armWidth=2,showVector=False):
        data=self.data

        imageHeight,imageWidth,channels=img.shape
        padding=30

        spacingX=(imageWidth-2*padding)/(len(data[0]))
        spacingY=(imageHeight-2*padding)/(len(data)-1)

        armLength=min(spacingX,spacingY)/3
        offsets=[
            np.array([armLength*math.cos(-5*math.pi/6),armLength*math.sin(-5*math.pi/6)]),
            np.array([armLength*math.cos(-math.pi/6),armLength*math.sin(-math.pi/6)]),
            np.array([0,0]),
            np.array([armLength*math.cos(math.pi/2),armLength*math.sin(math.pi/2)])]

        ys=np.linspace(padding,imageWidth-padding,len(data)-1)

        overlay=np.zeros_like(img,np.uint8)

        for (row,y) in enumerate(ys):
            xs=np.linspace(padding,len(data[row])*spacingX,len(data[row]))
            for (col,x) in enumerate(xs):
                island=data[row][col]

                #offset if in an offset row
                thisRowOffset=self.isRowOffset(row)
                if thisRowOffset:
                    x+=spacingX/2
                xy=np.array([x,y])

                #cv2.putText(img, str(self.getAvgAdjacentCharge(row,col,2))[0:4], (int(xy[0]),int(xy[1])), cv2.FONT_HERSHEY_COMPLEX, 0.4, BLACK)

                for (i,point) in enumerate(island):
                    if(point==1):
                        color=WHITE
                    elif(point==-1):
                        color=BLACK
                    else:
                        color=GREEN
                    thisxy=xy+offsets[i]
                    #cv2.circle(img,(int(thisxy[0]),int(thisxy[1])), int(spacingX/14), color, -1)



                    node=np.array([int(thisxy[0]),int(thisxy[1])])
                    center=np.array([int(xy[0]),int(xy[1])])

                    nearCenter=tuple(((4*center+node)/5).astype(int))
                    node=tuple(node)

                    if not simplifyIslands:
                        if showIslands:
                            #change arrow direction
                            if(color==WHITE):
                                point1=node
                                point2=nearCenter
                            else:
                                point2=node
                                point1=nearCenter

                            if(color==GREEN):
                                cv2.line(img,point1,point2,color,2)
                            else:
                                cv2.arrowedLine(img,point1,point2,color,armWidth,tipLength=spacingX/100)
                        else:
                            cv2.line(img,node,nearCenter,BLACK,1)

                if showIslands and simplifyIslands:
                    moment=getIslandMomentVector(island)
                    if moment is not None:
                        vector=np.array(moment)
                        start=tuple((center-vector*armLength/2).astype(int))
                        end=tuple((center+vector*armLength/2).astype(int))
                        cv2.arrowedLine(img,start,end,BLACK,armWidth,tipLength=spacingX/100)
                
                if showIslandCharge:
                    charge=self.getIslandCharge(row,col)
                    
                    if not halfInverted:
                        if(charge<0):color=BLUE
                        elif(charge>0):color=RED
                        else: color=GREEN
                    else:
                        if charge<0:color=ORANGE
                        elif charge>0:color=PURPLE
                        else: color=GREEN


                    coord=(int(xy[0]),int(xy[1]))
                    cv2.circle(img,coord,2+4*(abs(charge)),color,-1)


                
                if showVertexCharge:
                    neighbors=self.getNeighbors(row,col)

                    if(neighbors[BOTTOMLEFTNEIGHBOR] is not None and neighbors[BOTTOMRIGHTNEIGHBOR] is not None):
                        vertexCharges=self.getVerticies(row,col)


                        if halfInverted:
                            vertexCharges[2]*=-1

                        if not halfInverted:
                            if(vertexCharges[2]<0):color=BLUE
                            elif(vertexCharges[2]>0):color=RED
                            else: color=GREEN
                        else:
                            if vertexCharges[2]<0:color=ORANGE
                            elif vertexCharges[2]>0:color=PURPLE
                            else: color=GREEN

                        coord=(int(xy[0]+offsets[BOTTOM][0]),int(xy[1]+spacingX*math.sqrt(3)/3))
                        cv2.circle(img,coord,4+2*(abs(vertexCharges[2])),color,-1)

                if showRings:
                    #check if this island is the top left of a loop
                    if row<len(data)-2 and col<len(data[row])-2:
                        right=data[row][col+1]

                        if(thisRowOffset):
                            below=data[row+1][col+1]
                        else:
                            below=data[row+1][col]

                        ringxy=xy+np.array([spacingX/2,spacingY/3])

                        color=None
                        if(island[TOPRIGHT]==1 and right[TOPLEFT]==-1 and right[BOTTOM]==1 and below[TOPRIGHT]==-1 and below[TOPLEFT]==1 and island[BOTTOM]==-1):
                            color=RED
                        elif(island[TOPRIGHT]==-1 and right[TOPLEFT]==1 and right[BOTTOM]==-1 and below[TOPRIGHT]==1 and below[TOPLEFT]==-1 and island[BOTTOM]==1):
                            color=BLUE
                        if(color is not None):
                            cv2.circle(img,(int(ringxy[0]),int(ringxy[1])), int(spacingX/4), color, 2)
                
                if showVector and getIslandMomentVector(island) is not None:
                    vector=getIslandMomentVector(island)

                    start=np.array(xy)
                    end=start+vector*spacingX/3
                    start=start.astype(int)
                    end=end.astype(int)
                    #cv2.arrowedLine(img,start,end,RED,tipLength=0.2,thickness=2)
                    #cv2.putText(img,str(getIslandAngle(island))[0:3],coord,cv2.FONT_HERSHEY_COMPLEX,0.3,RED)
        
                    angle=restrictAngleDeg(getIslandAngle(island))
                    if angle is not None:
                        if twoColor:
                            if angle in [30,150,270]:
                                color=RED
                            else:
                                color=BLUE
                        else:
                            color=num_to_rgb(angle,max_val=360)
                            color=(color[0],color[1],color[2],0.1)
                        cv2.circle(overlay,tuple(start),int(spacingX/2),color,-1)

        alpha=0.5
        mask=overlay.astype(bool)
        img[mask]=cv2.addWeighted(img,alpha,overlay,1-alpha,0)[mask]

    def getChargeGrid(self):
        #WARNING: has some duplicate code with getVectorGrid

        vSpacing=math.sqrt(3)/2

        chargeGrid=ChargeGrid()

        for (y,row) in enumerate(self.data):
            thisRowOffset=self.isRowOffset(y)
            if thisRowOffset:
                xOffset=0.5
            else:
                xOffset=0

            for (x,col) in enumerate(row):
                
                islandCharge=self.getIslandCharge(y,x)
                if islandCharge!=0:
                    chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing,islandCharge,type="a"))

                vertexCharge=self.getVerticies(y,x)[2]
                neighbors=self.getNeighbors(y,x)

                if(vertexCharge!=0 and neighbors[BOTTOMLEFTNEIGHBOR] is not None and neighbors[BOTTOMRIGHTNEIGHBOR] is not None):
                    chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing+math.sqrt(3)/3,vertexCharge,type="b"))

        return chargeGrid

    def getMomentGrid(self):
        #WARNING: has some duplicate code with getCHargeGrid
        vSpacing=math.sqrt(3)/2

        moments=MomentGrid()

        for (y,row) in enumerate(self.data):
            thisRowOffset=self.isRowOffset(y)
            if thisRowOffset:
                xOffset=0.5
            else:
                xOffset=0

            for (x,island) in enumerate(row):
                realX=x+xOffset
                realY=y*vSpacing

                if not isBadIsland(island) and getIslandAngleRad(island) is not None:

                    angle=getIslandAngleRad(island)

                    moments.addMoment(Moment(realX,realY,angle,1))

        

        return moments

    def getLegMomentGrid(self):
        vSpacing=math.sqrt(3)/2
        hSpacing=1
        sideLength=math.sqrt(1/3)
        moments=MomentGrid()


        for (y,row) in enumerate(self.data):
            thisRowOffset=self.isRowOffset(y)
            if thisRowOffset:
                xOffset=0.5
            else:
                xOffset=0

            for (x,island) in enumerate(row):
                if isBadIsland(island):
                    continue

                islandX=x+xOffset
                islandY=y*vSpacing

                leftAngle=1*math.pi/6
                if island[0]==-1: leftAngle+=math.pi
                leftX=islandX-sideLength/2*R3O2
                leftY=islandY-sideLength/2*0.5

                rightAngle=5*math.pi/6
                if island[1]==-1: rightAngle+=math.pi
                rightX=islandX+sideLength/2*R3O2
                rightY=islandY-sideLength/2*0.5

                bottomAngle=3*math.pi/2
                if island[3]==-1: bottomAngle+=math.pi
                bottomX=islandX
                bottomY=islandY+sideLength/2


                moments.addMoment(Moment(leftX,leftY,leftAngle,1))
                moments.addMoment(Moment(rightX,rightY,rightAngle,1))
                moments.addMoment(Moment(bottomX,bottomY,bottomAngle,1))

        #print(moments.getGroupedMomentsByDistance(10, 9.814954576223638)[0:5])
        return moments
        



def chargeOrdering(y):
    cg=y.getChargeGrid()
    overall=cg.getOverallChargeCorrelation()
    overall=overall[1:9]
    print([el for el in overall])
    plt.pyplot.plot(range(len(overall)),[el[1] for el in overall])
    plt.pyplot.axhline(y=0, linestyle='-')

    plt.pyplot.show()

def drawDomains(lattice,outImg):
    lattice.draw(outImg,showVector=True,simplifyIslands=True)
    #lattice.draw(outImg,showVertexCharge=False,showIslandCharge=True,halfInverted=True)


def renderFile(lattice,outFolder):
    
    blankImg=np.zeros((1000,1000,3), np.uint8)
    blankImg[:,:]=(150,150,150)

    chargeImg=blankImg.copy()
    chargeImgHalfInverted=blankImg.copy()
    domainImg=blankImg.copy()
    twoColorDomainImg=blankImg.copy()
    ringImg=blankImg.copy()
    dipoleImg=blankImg.copy()

    drawDomains(lattice,domainImg)
    lattice.draw(twoColorDomainImg,showVector=True,simplifyIslands=True,twoColor=True)
    lattice.draw(chargeImg,showIslandCharge=True,showVertexCharge=True)
    lattice.draw(chargeImgHalfInverted,showIslandCharge=True,showVertexCharge=True,halfInverted=True)
    lattice.draw(ringImg,showRings=True)


    blankImg=np.zeros((2000,2000,3), np.uint8)
    blankImg[:,:]=(150,150,150)
    legMoments=lattice.getLegMomentGrid()
    legMoments.draw(dipoleImg)

    os.makedirs(outFolder,exist_ok=True)
    outFilePrefix=os.path.join(outFolder,fileName.split("/")[-1].split(".")[0][0:])
    
    cv2.imwrite(outFilePrefix+"_domains.jpg",domainImg)
    cv2.imwrite(outFilePrefix+"_charge.jpg",chargeImg)
    cv2.imwrite(outFilePrefix+"_chargeImgHalfInverted.jpg",chargeImgHalfInverted)
    cv2.imwrite(outFilePrefix+"_rings.jpg",ringImg)
    cv2.imwrite(outFilePrefix+"_twoColorDomains.jpg",twoColorDomainImg)
    cv2.imwrite(outFilePrefix+"_dipoles.jpg",dipoleImg)
    

    #cv2.imshow("window",domainImg)
    #cv2.waitKey(0)


outFolder="yShapeOut"

def getUserInputFiles():
    global outFolder

    parser = argparse.ArgumentParser(description='Y shaped lattice csv reader')
    parser.add_argument("file", type=str, help="Path to csv file or folder")
    parser.add_argument("--out", type=str, default="yShapeOut")
    args=parser.parse_args()

    outFolder=args.out

    fileNames=[]
    if os.path.isfile(args.file):
        fileNames=[args.file]
    elif os.path.isdir(args.file):
        for (dirpath, dirnames, files) in os.walk(args.file):
            for name in files:
                fileNames.append(os.path.join(dirpath,name))

    else:
        raise Exception("path is not file or directory")

    return fileNames

def assertChargeCorrelationGroupIsConsistent(chargeCorrelations):
    firstRow=chargeCorrelations[list(chargeCorrelations.keys())[0]]
    for i in chargeCorrelations.values():
        assert len(i)==len(firstRow)
    for columnIndex in range(0,len(firstRow)):
        expectedDistance=firstRow[columnIndex][0]
        for chargeCorrelation in chargeCorrelations.values():
            assert(abs(chargeCorrelation[columnIndex][0]-expectedDistance)<0.000001)
def writeCorrelationsToFile(chargeCorrelations,f):

    distances=[str(n[0]) for n in list(chargeCorrelations.values())[0]]
    f.write(f"distance=, {', '.join(distances)}\n")
    for fileName, chargeCorrelation in chargeCorrelations.items():
        f.write(fileName+", ")
        for pair in chargeCorrelation:
            f.write(str(pair[1])+", ")

        f.write("\n")
    f.write("\n\n")

if __name__=="__main__":
    if True:
        GEN_CHARGE_CORRELATION=False
        RENDER_FILES=False
        RUN_CORRELATION_VS_DISTANCE=False
        RUN_CORRELATION_VS_OFFSET=False
        RUN_CORRELATION_VS_ANGLE=False
        RUN_RELATIVE_ANGLE_VS_DISTANCE=False
        RUN_DIPOLE_CORRELATION=False
        RUN_DELTA_NU_CORRELATION=True


        fileNames=sorted(getUserInputFiles())

        
        chargeCorrelations={}
        AAChargeCorrelations={}
        BBChargeCorrelations={}
        ABChargeCorrelations={}

        correlationVsOffset={}
        correlationVsDistance={}
        dipoleCorrelation={}
        correlationVsAngle={}
        relativeAngleVsDistance={}
        deltaNuCorrelation={}

        for fileName in fileNames:
            print(f"analyzing: {fileName}")
            
            try:
                file=open(fileName,newline="\n")
            except:
                raise Exception("Error with file: "+str(fileName))

            data=getFileData(file)
            lattice=YShapeLattice(data)


            moments=lattice.getMomentGrid()

            if RUN_DIPOLE_CORRELATION:
                legMoments=lattice.getLegMomentGrid()

                dipoleCorrelation[fileName]=legMoments.getCorrelationVsDistance(7)

            if RUN_DELTA_NU_CORRELATION:
                legMoments=lattice.getLegMomentGrid()
                delta,nu=legMoments.getDeltaNuCorrelation(1)

                deltaNuCorrelation[fileName]={"delta":delta,"nu":nu}





            if RUN_CORRELATION_VS_OFFSET:
                correlationVsOffset[fileName]=moments.getCorrelationByOffsetAngle()

            if RUN_CORRELATION_VS_DISTANCE:
                correlationVsDistance[fileName]=moments.getCorrelationVsDistance(7)

            if RUN_CORRELATION_VS_ANGLE:
                correlationVsAngle[fileName]=moments.getCorrelationVsAbsoluteAngle(7)

            if RUN_RELATIVE_ANGLE_VS_DISTANCE:
                relativeAngleVsDistance[fileName]=moments.getRelativeAngleVsDistance(7)

            if RENDER_FILES:
                renderFile(lattice,outFolder)

                cg=lattice.getChargeGrid()

                blankImg=np.zeros((1000,1000,3), np.uint8)
                blankImg[:,:]=(150,150,150)
                cg.draw(blankImg,colorByType=False)
                #cv2.imwrite("test"+fileName[-7:-4]+".png",blankImg)

            if GEN_CHARGE_CORRELATION:
                chargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5)
                AAChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="a" and j.type=="a")
                ABChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="a" and j.type=="b")
                BBChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="b" and j.type=="b")
                

        if GEN_CHARGE_CORRELATION:
            assertChargeCorrelationGroupIsConsistent(chargeCorrelations)
            assertChargeCorrelationGroupIsConsistent(AAChargeCorrelations)
            assertChargeCorrelationGroupIsConsistent(ABChargeCorrelations)
            assertChargeCorrelationGroupIsConsistent(BBChargeCorrelations)



            with open(outFolder+"/chargeCorrelation.csv", "w") as f:
                
                f.write("Overall Correlation\n")
                writeCorrelationsToFile(chargeCorrelations,f)
                f.write("AACorrelation\n")
                writeCorrelationsToFile(AAChargeCorrelations,f)
                f.write("AB Correlation\n")
                writeCorrelationsToFile(ABChargeCorrelations,f)
                f.write("BB Correlation\n")
                writeCorrelationsToFile(BBChargeCorrelations,f)
                
                f.close()
    

        if RUN_CORRELATION_VS_OFFSET:
            possibleCenterAngles=[30,90,150,210,270,330]
            possibleOffsetAngles=[0,60,120,180,240,300]

            with open(outFolder+"/correlationVsOffset.csv", "w") as f:
                f.write(f"file name, center moment angle, {str(possibleOffsetAngles)[1:-1]}\n")

                for fileName,value in correlationVsOffset.items():
                    for centerMomentAngle in possibleCenterAngles:
                        f.write(f"{fileName}, {centerMomentAngle}, ")
                        for offsetAngle in possibleOffsetAngles:
                            f.write(str(value[centerMomentAngle][offsetAngle])+", ")
                        f.write("\n")

        if RUN_CORRELATION_VS_ANGLE:
            with open(outFolder+"/correlationVsAngle.csv", "w") as f:
                f.write("file name, distance, angle1, correlation1, angle2, correlation2, angle3, correlation3, ...\n")
                for fileName,values in sorted(correlationVsAngle.items()):
                    for n in range(1,len(values)):
                        f.write(fileName+", "+str(n)+", ")
                        for angle in sorted(values[n].keys()):
                            f.write(str(angle)+", "+str(values[n][angle])+", ")


                        f.write("\n")
        
        if RUN_CORRELATION_VS_DISTANCE:
            with open(outFolder+"/correlationVsDist.csv","w") as f:
                f.write(f"file, n=1, n=2, n=3, ...\n")

                for fileName,values in correlationVsDistance.items():
                    f.write(f"{fileName}, ")
                    for value in values[1:]:
                        f.write(f"{value}, ")
                    f.write("\n")
        
        if RUN_DIPOLE_CORRELATION:
            with open(outFolder+"/dipoleCorrelation.csv","w") as f:
                f.write(f"file, n=1, n=2, n=3, ...\n")

                for fileName,values in dipoleCorrelation.items():
                    f.write(f"{fileName}, ")
                    for value in values[1:]:
                        f.write(f"{value}, ")
                    f.write("\n")

        if RUN_DELTA_NU_CORRELATION:
            with open(outFolder+"/deltaNuCorrelation.csv","w") as f:
                f.write("file, delta, nu\n")
                for fileName,data in deltaNuCorrelation.items():
                    f.write(f"{fileName}, ")
                    f.write(f"{data['delta']},{data['nu']}")
                    f.write("\n")

        if RUN_RELATIVE_ANGLE_VS_DISTANCE:
            angles=[0,60,120,180,240,300]
            with open(outFolder+"/relativeAngleVsDist.csv","w") as f:
                f.write(f"file name, n, 0,60,120,180,240,300\n")
                for fileName, value in relativeAngleVsDistance.items():
                    print(relativeAngleVsDistance[fileName])
                    for n in range(len(relativeAngleVsDistance[fileName])):
                        f.write(f"{fileName}, {n}, ")
                        for angle in angles:
                            f.write(f"{relativeAngleVsDistance[fileName][n][angle]}, ")
                        f.write("\n")
            
                    

    else:
        fileNames=sorted(getUserInputFiles())
        for fileName in fileNames:
            try:
                file=open(fileName,newline="\n")
            except:
                raise Exception("Error with file: "+str(fileName))

            data=getFileData(file)
            lattice=YShapeLattice(data)
            
            moments=lattice.getMomentGrid()


            blankImg=np.zeros((1000,1000,3), np.uint8)
            blankImg[:,:]=(150,150,150)
            moments.draw(blankImg)

            print(moments.getCorrelationVsDistance(20))


