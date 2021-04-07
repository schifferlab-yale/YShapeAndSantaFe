import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
import csv


parser = argparse.ArgumentParser(description='Santa fe csv analysis')
parser.add_argument("file", type=str, help="Path to csv file")
args=parser.parse_args()


#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)


#just shifts every element in an array by one and loops the last one back to the front
def cyclicRotate(input):
     return ([input[-1]] + input[0:-1])

#inverts all numbers in an array and keeps None values
def invert(input):
    return [-i if i is not None else None for i in input]

class Cell:
    def __init__(self,arrow):
        assert arrow in ["up","left","down","right", None]
        self.arrow=arrow;
        self.center=False
        self.dimer=False
        self.interiorCenter=False
    def __str__(self):
        return str(self.arrow)

    @property
    def hasArrow(self):
        return self.arrow is not None


class SantaFeLattice:
    def __init__(self,csvFile):

        csvreader= csv.reader(file, delimiter=",")
        firstLine=next(csvreader)[0]
        if firstLine == "first row horizontal":
            firstRowHorizontal=True
        elif firstLine == "first row vertical":
            firstRowHorizontal=False
        else:
            raise Exception("CSV file did not specify whether first row was horizontal or vertical")

        self.updateRawData(csvreader)
        self.updateArrowData(firstRowHorizontal)
        self.updateVertexClassifactionDictionary()
        self.updateDimers()
        self.updateCenters()


        self.imagePaddingConstant=0.05

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
    def updateArrowData(self,firstRowHorizontal):
        rawData=self.rawData
        data=[]
        horizontal=firstRowHorizontal
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
                else:
                    row.append(Cell(None))

            data.append(row)
            horizontal=not horizontal
        self.data=data
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
        self.vertexClasses=verticies

    def getSpacingXSpacingY(self,image):
        imageHeight,imageWidth,imageChannels=image.shape

        spacingX=(imageWidth-2*self.imagePaddingConstant*imageWidth)/(self.colCount-1)
        spacingY=(imageHeight-2*self.imagePaddingConstant*imageHeight)/(self.rowCount-1)

        return spacingX,spacingY
    def getXYFromRowCol(self,row,col,image):
        imageHeight,imageWidth,imageChannels=image.shape

        paddingY=self.imagePaddingConstant*imageHeight
        paddingX=self.imagePaddingConstant*imageWidth

        y=paddingY+row/self.rowCount*(imageHeight-2*paddingY)
        x=paddingX+col/self.colCount*(imageWidth-2*paddingX)
        return (int(x),int(y))

    @property
    def rowCount(self):
        return len(self.data)
    @property
    def colCount(self):
        return len(self.data[0])

    def drawCells(self,image):
        data=self.data
        #make the line mesh
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                self.drawCell(rowI,colI,image)
    def drawCell(self,rowI,colI,image):
        cell=self.getCell(rowI,colI)
        x, y = self.getXYFromRowCol(rowI,colI,image)
        spacingX,spacingY=self.getSpacingXSpacingY(image)

        if(cell.arrow is not None):

            right=(int(x+spacingX/4),y)
            left=(int(x-spacingX/4),y)

            down=(x,int(y+spacingY/4))
            up=(x,int(y-spacingY/4))

            color=BLACK
            tipLength=0.5

            if(cell.arrow=="up"):
                image=cv2.arrowedLine(image,down,up,color,1,tipLength=tipLength)
            elif(cell.arrow=="down"):
                image=cv2.arrowedLine(image,up,down,color,1,tipLength=tipLength)
            elif(cell.arrow=="left"):
                image=cv2.arrowedLine(image,right,left,color,1,tipLength=tipLength)
            elif(cell.arrow=="right"):
                image=cv2.arrowedLine(image,left,right,color,1,tipLength=tipLength)
        elif(cell.dimer):
            images=cv2.circle(image,(x,y),2,RED,-1)
        elif(cell.center):
            images=cv2.circle(image,(x,y),2,BLUE,-1)

        #image=cv2.putText(image, str(self.countInOutArrows(rowI,colI)), (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.3, BLACK, 1, cv2.LINE_AA)

    def getCell(self,rowI,colI):
        if(rowI>=0 and rowI<len(self.data) and colI>=0 and colI<len(self.data[rowI])):
            return self.data[rowI][colI]
        else:
            return None

    def getNeighbors(self,rowI,colI):

        top=self.getCell(rowI-1,colI)
        bottom=self.getCell(rowI+1,colI)

        left=self.getCell(rowI,colI-1)
        right=self.getCell(rowI,colI+1)

        return top,right,bottom,left

    def countInOutArrows(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        count=0
        if(top is not None and (top.arrow=="up" or top.arrow=="down")):
            count+=1
        if(bottom is not None and (bottom.arrow=="up" or bottom.arrow=="down")):
            count+=1
        if(right is not None and (right.arrow=="right" or right.arrow=="left")):
            count+=1
        if(left is not None and (left.arrow=="right" or left.arrow=="left")):
            count+=1
        return count

    def countArrowNeighbors(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        all=[top,right,bottom,left]
        count=0
        for island in all:
            if island is not None and island.hasArrow:
                count+=1
        return count

    def isCenter(self,rowI,colI):
        if(self.countArrowNeighbors(rowI,colI)==0):
            return True
    def isInteriorCenter(self,rowI,colI):
        if(not self.getCell(rowI,colI).center):
            raise Exception("This function should only be called on centers")

        potentialCenters=[(rowI+2,colI),(rowI-2,colI),(rowI,colI+2),(rowI+2,colI)]
        raise Exception("you were here")
        for cell in potentialCenters:
            if(cell.center):
                return True


    def getInOutPattern(self,rowI,colI):
        top,right,bottom,left=self.getNeighbors(rowI,colI)
        pattern=[]
        if top is not None:
            if(top.arrow=="down"):
                pattern.append(1)
            elif(top.arrow=="up"):
                pattern.append(-1)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if right is not None:
            if(right.arrow=="left"):
                pattern.append(1)
            elif(right.arrow=="right"):
                pattern.append(-1)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if bottom is not None:
            if(bottom.arrow=="up"):
                pattern.append(1)
            elif(bottom.arrow=="down"):
                pattern.append(-1)
            else:
                pattern.append(None)
        else:
            pattern.append(None)
        if left is not None:
            if(left.arrow=="right"):
                pattern.append(1)
            elif(left.arrow=="left"):
                pattern.append(-1)
            else:
                pattern.append(None)
        else:
            pattern.append(None)

        return pattern


    def isDimer(self, rowI,colI):
        cell=self.getCell(rowI,colI)

        vertexPattern=self.getInOutPattern(rowI,colI)
        islandCount=self.countInOutArrows(rowI,colI)

        vertexType=0;

        if islandCount>1 and cell.arrow is None:
            #get vertex type
            for type in self.vertexClasses[islandCount].keys():

                if vertexPattern in self.vertexClasses[islandCount][type]:
                    vertexType=type
                    break
        return vertexType>1

    def updateDimers(self):
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                cell.dimer=self.isDimer(rowI,colI)

    def updateCenters(self):
        data=self.data
        for (rowI, row) in enumerate(data):
            for(colI, cell) in enumerate(row):
                cell.center=self.isCenter(rowI,colI)


if __name__=="__main__":
    try:
        file=open(args.file,newline="\n")
    except:
        raise Exception("Error with file")
    lattice=SantaFeLattice(file)

    outputImage=np.zeros((1000,1000,3), np.uint8)
    outputImage[:,:]=(250,250,250)
    lattice.drawCells(outputImage)


    cv2.imshow("window",outputImage)
    cv2.waitKey(0)
