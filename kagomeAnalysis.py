import argparse
from os import close
import cv2
import numpy as np
import math


#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)
ORANGE=(0,165,255)
PURPLE=(130,0,75)

def getFile():
    parser = argparse.ArgumentParser(description='Kagome lattice csv reader')
    parser.add_argument("file", type=str, help="Path to csv")
    args=parser.parse_args()
    return args.file

def fileToData(fileName):
    try:
        file=open(fileName,newline="\n")
    except:
        raise Exception("Error with file: "+str(fileName))

    file=file.read()

    file=file.replace("\r","")#some systems use \r\n, others use \n
    file=file.split("\n")

    file=[line.split(",") for line in file]

    data=[]
    for line in file:
        newLine=[]
        for char in line:
            if char==' ' or char=="\t" or char=="  " or char=='':
                newLine.append(None)
            else:
                newLine.append(int(char))
        data.append(newLine)

    return data

def isRowCloselySpaced(row):
    numNone=0
    for el in row:
        if el is None:
            numNone+=1

    if numNone-1<len(row)/2:
        return True
    return False


class KagomeLattice():
    def __init__(self,data):
        self.data=data
        self.determineRowType()

    def determineRowType(self):
        evenCount=[0,0]
        for (i,row) in enumerate(self.data):
            if isRowCloselySpaced(row):
                evenCount[i%2]+=1

        if evenCount[0]>evenCount[1]:
            self.firstRowCloselySpaced=True
        else:
            self.firstRowCloselySpaced=False

    def coordInLattice(self,row,col):
        return row>0 and col>0 and row<len(self.data) and col<len(self.data[row])
    def cellValueOrNone(self,row,col):
        if self.coordInLattice(row,col):
            return self.data[row][col]
            
        return None
    def determineAngle(self,rowI,colI):

        closelySpaced=not (self.firstRowCloselySpaced ^ rowI%2==0)
        if not closelySpaced:
            return 90
        else:
            if self.cellValueOrNone(rowI-1,colI-1) is not None:
                angle=-45
            elif self.cellValueOrNone(rowI-1,colI+1) is not None:
                angle=45
            elif self.cellValueOrNone(rowI+1,colI-1) is not None:
                angle=45
            elif self.cellValueOrNone(rowI+1,colI+1) is not None:
                angle=-45

        return angle



    def draw(self,im,border=50):
        data=self.data

        lineRadius=6
        lineWidth=5
        r2radius=round(lineRadius*math.sqrt(2)/2)

        for (rowI,row) in enumerate(data):
            y=int(np.interp(rowI,[0,len(data)],[border,len(img)-border]))
            closelySpaced=not (self.firstRowCloselySpaced ^ rowI%2==0)
            for (colI,cell) in enumerate(row):
                x=int(np.interp(colI,[0,len(row)],[border,len(img[y])-border]))
                if cell is not None and (cell==1 or cell==-1):
                    
                    xy=np.array([x,y])

                    if cell==1:
                        color=BLACK
                    elif cell==-1:
                        color=WHITE


                    """
                    if not closelySpaced:
                        cv2.line(im,(x,y-lineRadius),(x,y+lineRadius),color,lineWidth)
                        
                    else:
                        angle=self.determineAngle(rowI,colI)
                        if angle==45:
                            cv2.line(im,(x-r2radius,y+r2radius),(x+r2radius,y-r2radius),color,lineWidth)
                        elif angle==-45:
                            cv2.line(im,(x-r2radius,y-r2radius),(x+r2radius,y+r2radius),color,lineWidth)
                        else:
                            cv2.circle(im,(x,y),2,color,-1)
                    """
                    angle=self.determineAngle(rowI,colI)
                    vector=np.array([round(2*lineRadius*math.cos(angle/180*math.pi)),-round(2*lineRadius*math.sin(angle/180*math.pi))])
                    if cell==-1:
                        vector*=-1
                    cv2.arrowedLine(im,xy+vector,xy-vector,color,lineWidth,tipLength=0.2)
                    #cv2.line(im,xy+vector,xy-vector,color,lineWidth)
                else: 
                    cv2.circle(im,(x,y),1,BLACK,-1)








if __name__ == "__main__":
    file=getFile()
    data=fileToData(file)
    k=KagomeLattice(data)

    img=np.zeros((1000,1000,3),np.uint8)
    img[:,:]=(200,200,200)
    k.draw(img)
    cv2.imshow("window",img)
    cv2.waitKey(0)

