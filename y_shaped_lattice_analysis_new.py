import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
import csv
import matplotlib as plt

from numpy.lib.function_base import average

#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)


TOPLEFT=0
TOPRIGHT=1
MIDDLE=2
BOTTOM=3

TOPLEFTNEIGHBOR=0
TOPRIGHTNEIGHBOR=1
LEFTNEIGHBOR=2
RIGHTNEIGHBOR=3
BOTTOMLEFTNEIGHBOR=4
BOTTOMRIGHTNEIGHBOR=5

TOPVERTEX=0
LEFTVERTEX=1
RIGHTVERTEX=2

def getFileData(file):
    file=file.read().replace("\t","")
    file=file.split("\r\n")
    file=[line.split(", ") for line in file]
    return file

class Charge:
    def __init__(self,x,y,charge):
        self.x=x
        self.y=y
        self.charge=charge
    def __repr__(self):
        return f"{self.charge}"
        return f"{self.charge} at {self.x},{self.y}"

class ChargeGrid:
    def __init__(self):
        self.charges=[]
    def addCharge(self,charge):
        self.charges.append(charge)
    def draw(self,img,padding=50):

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

            if(charge.charge<0):color=BLACK
            elif(charge.charge>0):color=WHITE
            else: color=GREEN

            cv2.circle(img,(int(imgX),int(imgY)),2*abs(charge.charge), color,-1)

    def chargesByDistance(self,x,y):
        out=[]
        for charge in self.charges:
            dist=math.sqrt((charge.x-x)**2+(charge.y-y)**2)
            out.append((dist,charge))

        out=sorted(out,key=lambda el:el[0])
        return out
    def getGroupedChargesByDistance(self,x,y):
        error=0.00001#for floating point error in distances

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
    
    def getAllCorrelationPairs(self):
        pairs=[]
        for charge in self.charges:
            for otherCharge in self.charges:
                dist=math.sqrt((charge.x-otherCharge.x)**2+(charge.y-otherCharge.y)**2)
                correlation=charge.charge*otherCharge.charge
                pairs.append((dist,correlation))
        return pairs
    
    def getOverallChargeCorrelation(self):
        correlations=self.getAllCorrelationPairs()
        correlations=sorted(correlations,key=lambda el:el[0])

        error=0.00000001
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

        topLeft=neighbors[TOPLEFTNEIGHBOR][BOTTOM]+neighbors[LEFTNEIGHBOR][TOPRIGHT]+thisCell[TOPLEFT]
        topRight=neighbors[TOPRIGHTNEIGHBOR][BOTTOM]+neighbors[RIGHTNEIGHBOR][TOPLEFT]+thisCell[TOPRIGHT]
        bottom=neighbors[BOTTOMLEFTNEIGHBOR][TOPRIGHT]+neighbors[BOTTOMRIGHTNEIGHBOR][TOPLEFT]+thisCell[BOTTOM]

        return [topLeft,topRight,bottom]
    
    def getIslandCharge(self,row,col):
        island=self.data[row][col]

        return -(island[TOPLEFT]+island[TOPRIGHT]+island[BOTTOM])

    def draw(self,img,showIslands=True, showIslandCharge=True, showRings=False,showVertexCharge=False):
        data=self.data

        imageHeight,imageWidth,channels=img.shape
        padding=100

        spacingX=(imageWidth-2*padding)/(len(data[0])-1)
        spacingY=(imageHeight-2*padding)/(len(data)-1)

        armLength=min(spacingX,spacingY)/2.5
        offsets=[
            np.array([armLength*math.cos(-5*math.pi/6),armLength*math.sin(-5*math.pi/6)]),
            np.array([armLength*math.cos(-math.pi/6),armLength*math.sin(-math.pi/6)]),
            np.array([0,0]),
            np.array([armLength*math.cos(math.pi/2),armLength*math.sin(math.pi/2)])]

        ys=np.linspace(padding,imageWidth-padding,len(data)-1)
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
                            cv2.arrowedLine(img,point1,point2,color,2,tipLength=spacingX/400)
                    else:
                        cv2.line(img,node,nearCenter,BLACK,1)

                
                if showIslandCharge:
                    charge=self.getIslandCharge(row,col)
                    if charge<0:color=BLACK
                    elif charge>0: color=WHITE
                    else: color=GREEN

                    coord=(int(xy[0]),int(xy[1]))
                    cv2.circle(img,coord,2*(abs(charge)),color,-1)


                
                if showVertexCharge:
                    if(row==len(ys)-1):
                        pass#skip last row
                    else:
                        vertexCharges=self.getVerticies(row,col)

                        if(vertexCharges[2]<0):color=BLACK
                        elif(vertexCharges[2]>0):color=WHITE
                        else: color=GREEN

                        coord=(int(xy[0]+offsets[BOTTOM][0]),int(xy[1]+offsets[BOTTOM][1]+spacingY/5))
                        cv2.circle(img,coord,2*(abs(vertexCharges[2])),color,-1)

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

    

    def getChargeGrid(self):

        vSpacing=math.sqrt(3)/2

        chargeGrid=ChargeGrid()

        for (y,row) in enumerate(self.data):

            
            thisRowOffset=self.isRowOffset(y)
            if thisRowOffset:
                xOffset=0.5
            else:
                xOffset=0

            for (x,col) in enumerate(row):
                
                chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing,self.getIslandCharge(y,x)))

                vertexCharge=self.getVerticies(y,x)[2]
                if(x!=0 and x!=len(row)-1):
                    chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing+math.sqrt(3)/3,vertexCharge))

        return chargeGrid


    """
    def getAdjacentCharges(self,row,col,level):
        island=self.data[row][col]

        if level==1:
            return self.getVerticies(row,col)
        elif level==2:
            neighborCoords=self.getNeighborsCoords(row,col)
            neighborCoords=[el for el in neighborCoords if el is not None]
            return [self.getIslandCharge(*coord) for coord in neighborCoords]
        elif level==3:
            charges=[]
            charges.append()
        else:
            raise NotImplementedError();
    
    def getAvgAdjacentCharge(self,row,col,level):
        return np.average(self.getAdjacentCharges(row,col,level))"""



if __name__=="__main__":
    if False:
        parser = argparse.ArgumentParser(description='Y shaped lattice csv reader')
        parser.add_argument("file", type=str, help="Path to csv file")
        args=parser.parse_args()

        try:

            file=open(args.file,newline="\n")
        except:
            raise Exception("Error with file")
        

        data=getFileData(file)

        y=YShapeLattice(data)

        outputImage1=np.zeros((1000,1000,3), np.uint8)
        outputImage1[:,:]=(150,150,150)

        y.draw(outputImage1,showVertexCharge=True,showIslands=True,showIslandCharge=True)

        chargeImg=np.zeros((1000,1000,3), np.uint8)
        chargeImg[:,:]=(150,150,150)

        cg=y.getChargeGrid()
        cg.draw(chargeImg)
        print(cg.getChargeCorrelation(cg.charges[2]))
        overall=cg.getOverallChargeCorrelation()
        overall=overall[1:10]
        print([el for el in overall])
        plt.pyplot.plot(range(len(overall)),[el[1] for el in overall])

        plt.pyplot.show()

        #cv2.imshow("grid",chargeImg)

        #cv2.imshow("window",outputImage1)
        #cv2.waitKey(0)

        #cv2.imwrite("analysis-output.jpg", np.float32(outputImage));
    else:
        files=[
            '.\Y-shape(6-14-21)\\107.csv',
            '.\Y-shape(6-14-21)\\108.csv',
            '.\Y-shape(6-14-21)\\109.csv',
            '.\Y-shape(6-14-21)\\110.csv',
            '.\Y-shape(6-14-21)\\111.csv',
            '.\Y-shape(6-14-21)\\112.csv',
            '.\Y-shape(6-14-21)\\113.csv',
            '.\Y-shape(6-14-21)\\114.csv',
            '.\Y-shape(6-14-21)\\115.csv',
            '.\Y-shape(6-14-21)\\116.csv',
            '.\Y-shape(6-14-21)\\117.csv',
            '.\Y-shape(6-14-21)\\118.csv'
        ]
        for fileName in files:
            file=open(fileName,newline="\n")

            data=getFileData(file)
            y=YShapeLattice(data)
            cg=y.getChargeGrid()

            overall=cg.getOverallChargeCorrelation()
            overall=overall[1:11]

            print(fileName,end=", ")
            for el in overall:
                print(f"{el[0]}",end=", ")
            for el in overall:
                print(f"{el[1]}",end=", ")
            print("\n")
