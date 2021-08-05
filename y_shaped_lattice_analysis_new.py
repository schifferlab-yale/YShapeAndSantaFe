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
    
    file=file.replace("\r\n","\n")#some systems use \r\n, others use \n
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
        return f"{self.charge}"
        return f"{self.charge} at {self.x},{self.y}"

#gets a numpy array [x,y] for the moment direction of an island
def getIslandMomentVector(island):
    topLeft=np.array([-R3O2,-0.5])*-island[TOPLEFT]
    topRight=np.array([R3O2,-0.5])*-island[TOPRIGHT]
    bottom=np.array([0,1])*-island[BOTTOM]
    return topLeft+topRight+bottom

#gets angle of island moment vector (in degrees)
def getIslandAngle(island):
    vector=getIslandMomentVector(island)
    angleRad=math.atan2(vector[1],vector[0])
    angle=int(angleRad/(2*math.pi)*360)
    return angle

#https://stackoverflow.com/questions/57400584/how-to-map-a-range-of-numbers-to-rgb-in-python
def num_to_rgb(val, max_val=1):
    i = (val * 255 / max_val);
    r = round(math.sin(0.024 * i + 0) * 127 + 128);
    g = round(math.sin(0.024 * i + 2) * 127 + 128);
    b = round(math.sin(0.024 * i + 4) * 127 + 128);
    return (r,g,b)

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

    def draw(self,img,showIslands=True, showIslandCharge=False, showRings=False,showVertexCharge=False, halfInverted=False,armWidth=2,showVector=False):
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
                
                if showVector:
                    vector=getIslandMomentVector(island)
                    start=np.array(xy)
                    end=start+vector*spacingX/3
                    start=start.astype(int)
                    end=end.astype(int)
                    #cv2.arrowedLine(img,start,end,RED,tipLength=0.2,thickness=2)
                    #cv2.putText(img,str(getIslandAngle(island))[0:3],coord,cv2.FONT_HERSHEY_COMPLEX,0.3,RED)
                    angle=getIslandAngle(island)
                    color=num_to_rgb(angle,max_val=360)
                    color=(color[0],color[1],color[2],0.1)


                    
                    cv2.circle(overlay,start,int(spacingX/2),color,-1)

        alpha=0.5
        mask=overlay.astype(bool)
        img[mask]=cv2.addWeighted(img,alpha,overlay,1-alpha,0)[mask]

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
                
                chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing,self.getIslandCharge(y,x),type="a"))

                vertexCharge=self.getVerticies(y,x)[2]

                neighbors=self.getNeighbors(y,x)

                if(neighbors[BOTTOMLEFTNEIGHBOR] is not None and neighbors[BOTTOMRIGHTNEIGHBOR] is not None):
                    chargeGrid.addCharge(Charge(x+xOffset,y*vSpacing+math.sqrt(3)/3,vertexCharge,type="b"))

        return chargeGrid



def chargeOrdering(y):
    cg=y.getChargeGrid()
    overall=cg.getOverallChargeCorrelation()
    overall=overall[1:9]
    print([el for el in overall])
    plt.pyplot.plot(range(len(overall)),[el[1] for el in overall])
    plt.pyplot.axhline(y=0, linestyle='-')

    plt.pyplot.show()

def drawDomains(lattice,outImg):
    lattice.draw(outImg,showVector=True)
    #lattice.draw(outImg,showVertexCharge=False,showIslandCharge=True,halfInverted=True)


def renderFile(lattice,outFolder):
    
    blankImg=np.zeros((1000,1000,3), np.uint8)
    blankImg[:,:]=(150,150,150)

    chargeImg=blankImg.copy()
    chargeImgHalfInverted=blankImg.copy()
    domainImg=blankImg.copy()

    drawDomains(lattice,domainImg)
    lattice.draw(chargeImg,showIslandCharge=True,showVertexCharge=True)
    lattice.draw(chargeImgHalfInverted,showIslandCharge=True,showVertexCharge=True,halfInverted=True)

    os.makedirs(outFolder,exist_ok=True)
    outFilePrefix=os.path.join(outFolder,fileName.split("/")[-1].split(".")[0][0:])
    
    cv2.imwrite(outFilePrefix+"_domains.jpg",domainImg)
    cv2.imwrite(outFilePrefix+"_charge.jpg",chargeImg)
    cv2.imwrite(outFilePrefix+"_chargeImgHalfInverted.jpg",chargeImgHalfInverted)
    

    #cv2.imshow("window",domainImg)
    #cv2.waitKey(0)



def getUserInputFiles():
    parser = argparse.ArgumentParser(description='Y shaped lattice csv reader')
    parser.add_argument("file", type=str, help="Path to csv file or folder")
    args=parser.parse_args()

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
    fileNames=getUserInputFiles()
    
    chargeCorrelations={}
    AAChargeCorrelations={}
    BBChargeCorrelations={}
    ABChargeCorrelations={}

    for fileName in fileNames[0:1]:
        print(f"analyzing: {fileName}")
        
        try:
            file=open(fileName,newline="\n")
        except:
            raise Exception("Error with file: "+str(fileName))

        data=getFileData(file)
        lattice=YShapeLattice(data)

        #renderFile(lattice,"yShapeOut")

        cg=lattice.getChargeGrid()

        blankImg=np.zeros((1000,1000,3), np.uint8)
        blankImg[:,:]=(150,150,150)
        cg.draw(blankImg,colorByType=True)
        cv2.imwrite("test.png",blankImg)

        chargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5)
        AAChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="a" and j.type=="a")
        ABChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="a" and j.type=="b")
        BBChargeCorrelations[fileName]=cg.getOverallChargeCorrelation(maxDist=5,shouldCompare=lambda i,j: i.type=="b" and j.type=="b")
    

    assertChargeCorrelationGroupIsConsistent(chargeCorrelations)



    with open("yShapeOut/chargeCorrelation.csv", "w") as f:
        
        f.write("Overall Correlation\n")
        writeCorrelationsToFile(chargeCorrelations,f)
        f.write("AACorrelation\n")
        writeCorrelationsToFile(AAChargeCorrelations,f)
        f.write("AB Correlation\n")
        writeCorrelationsToFile(ABChargeCorrelations,f)
        f.write("BB Correlation\n")
        writeCorrelationsToFile(BBChargeCorrelations,f)
        
        f.close()
    

        


    """else:
        files=[
            'Y-shape(7-7-21)/206.csv',
            'Y-shape(7-7-21)/207.csv',
            'Y-shape(7-7-21)/208.csv',
            'Y-shape(7-7-21)/209.csv',
            'Y-shape(7-7-21)/210.csv',
            'Y-shape(7-7-21)/211.csv',
            'Y-shape(7-7-21)/215.csv',
            'Y-shape(7-7-21)/216.csv',
            'Y-shape(7-7-21)/217.csv',
        ]
        for fileName in files:
            file=open(fileName,newline="\n")

            data=getFileData(file)
            y=YShapeLattice(data)
            cg=y.getChargeGrid()

            chargeImg=np.zeros((1000,1000,3), np.uint8)
            chargeImg[:,:]=(150,150,150)

            chargeImg_halfInverted=np.zeros((1000,1000,3), np.uint8)
            chargeImg_halfInverted[:,:]=(150,150,150)

            ringImg=np.zeros((1000,1000,3), np.uint8)
            ringImg[:,:]=(150,150,150)

            #cg.draw(chargeImg)

            y.draw(chargeImg,showIslands=True,showVertexCharge=True,halfInverted=False)
            y.draw(chargeImg_halfInverted,showIslands=True,showVertexCharge=True,halfInverted=True)
            y.draw(ringImg,showIslands=True,showRings=True,showIslandCharge=False,armWidth=2)

            #cv2.imshow("window",chargeImg)
            #print(fileName.split(".")[0][0:]+"_charge.jpg")
            cv2.imwrite(fileName.split(".")[0][0:]+"_charge.jpg",chargeImg)
            cv2.imwrite(fileName.split(".")[0][0:]+"_charge_halfInverted.jpg",chargeImg_halfInverted)
            cv2.imwrite(fileName.split(".")[0][0:]+"_rings.jpg",ringImg)
            #cv2.waitKey(0)


            if (True):#charge correlation
                overall=cg.getOverallChargeCorrelation()
                overall=overall[1:11]

                print(fileName,end=", ")
                for el in overall:
                    print(f"{el[0]}",end=", ")
                for el in overall:
                    print(f"{el[1]}",end=", ")
                print("\n",end="")"""
