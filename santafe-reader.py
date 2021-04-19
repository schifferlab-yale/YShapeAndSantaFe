import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
from nodeNetwork import *
import json

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
parser.add_argument('-r', "--rows",  help="number of rows", type=int, default=50)
parser.add_argument('-c', "--columns",  help="number of columns", type=int, default=50)
parser.add_argument('-s', "--spacing",  help="how dense the islands are packed small=denser (default=0.25)", type=float, default=0.25)
parser.add_argument("-a", "--reference_image", help="image of the height(to help line up the sample points)", type=str)
parser.add_argument("-o", "--offset",  help="flip rotation of the larger islands",action='store_true', default=False)

args=parser.parse_args()

try:
    image = cv2.imread(args.image[0])
    image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")

if args.reference_image is not None:
    try:
        height_image = cv2.imread(args.reference_image)
        height_image = cv2.resize(height_image, (1000,1000))
    except:
        raise Exception("File not found")
else:
    height_image=np.zeros((1000,1000,3), np.uint8)



#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)

offset=args.offset
rowOffset=0
colOffset=0


class SantaFeNodeNetwork(NodeNetwork):
    def getSamplePointsFromSquare(self,topLeft,topRight,bottomLeft,bottomRight,row=0,col=0):
        #multiplier for how far the sample points are from the edge of the square
        shiftConstant=0.25

        row+=rowOffset
        col+=colOffset

        samplePoints=[]

        sample=False


        sampleAreas=[
        [0,1,1,0,1,1,0,1,1,0,1,1,0],
        [1,0,0,1,0,0,0,0,0,1,0,0,1],
        [1,0,0,1,0,0,0,0,0,1,0,0,1],
        [0,0,0,0,1,1,0,1,1,0,0,0,0],
        [1,0,0,1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1,0,0,1],
        [0,1,1,0,0,0,0,0,0,0,1,1,0],
        [1,0,0,1,0,0,1,0,0,1,0,0,1],
        [1,0,0,1,0,0,1,0,0,1,0,0,1],
        [0,0,0,0,1,1,0,1,1,0,0,0,0],
        [1,0,0,1,0,0,0,0,0,1,0,0,1],
        [1,0,0,1,0,0,0,0,0,1,0,0,1],
        [0,1,1,0,1,1,0,1,1,0,1,1,0]
        ]

        isOdd=(int((row%24)/12)+int((col%24)/12))%2==1
        if isOdd and offset==False or not isOdd and offset==True:

            sampleAreas=np.rot90(np.array(sampleAreas))

        if(sampleAreas[row%12][col%12]==1):
            sample=True



        if(sample):
            center=[(topLeft[0]+topRight[0]+bottomLeft[0]+bottomRight[0])/4,(topLeft[1]+topRight[1]+bottomLeft[1]+bottomRight[1])/4]
            samplePoints.append(center)





        return samplePoints
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        surroundingPoints=[]
        if(rowI >0):
            surroundingPoints.append(samplePoints[rowI-1][vertexI])
        if(vertexI>0):
            surroundingPoints.append(samplePoints[rowI][vertexI-1])

        for point in surroundingPoints:
            if point==[] or point[0][2]==0 or samplePoints[rowI][vertexI][0][2]==0:
                continue
            elif point[0][2]==samplePoints[rowI][vertexI][0][2]:
                return True

        return (False)

    def dataAsString(self):
        string=""
        pure=False


        if(pure):
            for row in self.samplePoints:
                string+="\n"
                for cell in row:
                    if(cell==[]):
                        string+="0"
                    else:
                        string+=str(cell[0][2])
                    string+=","
        else:
            if(rowOffset%3==0):
                string+="first row horizontal\n"
            else:
                string+="first row vertical\n"

            for (rowI, row) in enumerate(self.samplePoints):
                rowI+=rowOffset
                for(colI, cell) in enumerate(row):
                    colI+=colOffset

                    if(rowI%3==0):
                        #we are in a flat row
                        if(colI%3==1):
                            if(cell==[]):
                                string+=" ,"
                            else:
                                string+=str(cell[0][2]*-1)+","
                        elif(colI%3==0):
                            string+=" ,"

                    elif(rowI%3==1):
                        #we are in the top of a vertical row
                        if(colI%3==0):
                            if(cell!=[]):
                                string+=str(cell[0][2]*-1)+","
                            else:
                                string+=" ,"
                        elif(colI%3==1):
                            string+=' ,'

                    else:
                        continue
                if(rowI%3!=2):
                    string+="\n"


        return string
    def drawData(self,im):
        if self.dragging:
            return


        samplePoints=self.samplePoints


        for (rowI,row) in enumerate(samplePoints):
            for(cellI,cell) in enumerate(row):

                surroundingPoints=[]
                if(rowI >0):
                    surroundingPoints.append(samplePoints[rowI-1][cellI])
                if(cellI>0):
                    surroundingPoints.append(samplePoints[rowI][cellI-1])

                for point in surroundingPoints:
                    if point==[] or cell==[]:
                        continue
                    else:
                        if(cell[0][2]==1):
                            im=cv2.arrowedLine(im,(int(cell[0][0]),int(cell[0][1])),(int(point[0][0]),int(point[0][1])),WHITE,1,tipLength=0.3)
                        else:
                            im=cv2.arrowedLine(im,(int(point[0][0]),int(point[0][1])),(int(cell[0][0]),int(cell[0][1])),WHITE,1,tipLength=0.3)

    def correctError(self,rowI,cellI):
        samplePoints=self.samplePoints
        cell=self.samplePoints[rowI][cellI]
        row=samplePoints[rowI]
        if(len(cell)==0):
            return#no point here

        otherRowI=None
        otherCellI=None
        if(rowI>0 and samplePoints[rowI-1][cellI]!=[]):
            otherRowI=rowI-1
            otherCellI=cellI
        if(cellI>0 and samplePoints[rowI][cellI-1]!=[]):
            otherRowI=rowI
            otherCellI=cellI-1
        if(rowI<len(samplePoints)-1 and samplePoints[rowI+1][cellI]!=[]):
            otherRowI=rowI+1
            otherCellI=cellI
        if(cellI<len(row)-1 and samplePoints[rowI][cellI+1]!=[]):
            otherRowI=rowI
            otherCellI=cellI+1

        if(otherRowI is None):#no surrounding points
            return

        if(cell[0][2]!=samplePoints[otherRowI][otherCellI][0][2]):
            return; #no error

        if cell[0][2]==0 or samplePoints[otherRowI][otherCellI][0][2]==0:
            return#bad data

        color=self.sampleImageColor(image,cell[0][0],cell[0][1])
        otherColor=self.sampleImageColor(image,samplePoints[otherRowI][otherCellI][0][0],samplePoints[otherRowI][otherCellI][0][1])

        if(color>otherColor):
            cell[0][2]=1
            samplePoints[otherRowI][otherCellI][0][2]=-1
        else:
            cell[0][2]=-1
            samplePoints[otherRowI][otherCellI][0][2]=1
    def correctErrors(self):
        samplePoints=self.samplePoints
        for (rowI,row) in enumerate(samplePoints):
            for(cellI,cell) in enumerate(row):
                self.correctError(rowI,cellI)






n=SantaFeNodeNetwork(Node(25,66),Node(551,66),Node(26,529),Node(545,529),args.rows, args.columns,image)


show_ref_image=False

def show():
    imWidth=1000;
    imHeight=1000;

    if(show_ref_image):
        outputImage=height_image.copy()
    else:
        outputImage=image.copy()

    n.draw(outputImage)
    cv2.imshow("window",outputImage)

    outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    outputImage[:,:]=BLACK
    n.drawData(outputImage)
    cv2.imshow("output",outputImage)





lastMouse=(0,0)
def mouse_event(event, x, y,flags, param):
    global lastMouse
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
"""
    print(n.topLeft)
    print(n.topRight)
    print(n.bottomLeft)
    print(n.bottomRight)
    print(n.rows)
    print(n.cols)"""


show();
cv2.setMouseCallback('window', mouse_event)

while(True):
    key=cv2.waitKey(0)
    if(key==ord("\r")):
        break;
    elif(key==ord("r")):
        n.addRow()
    elif(key==ord("e")):
        n.removeRow()

    elif(key==ord("c")):
        n.addCol()
    elif(key==ord("x")):
        n.removeCol()
    elif(key==ord("o")):
        offset=not offset
        n.setSamplePoints()
    elif(key==ord("q")):
        show_ref_image=not show_ref_image
    elif(key==ord("a")):
        rowOffset+=1
        n.setSamplePoints()
    elif(key==ord("b")):
        colOffset+=1
        n.setSamplePoints()
    elif(key==ord("j")):
        for i in range(10):
            n.jiggleNearestFixedPoint(*lastMouse)
            print(i)
        print("done")
    elif(key==ord("f")):
        n.correctErrors()
    elif(key==ord("g")):
        nearest=n.getNearestSamplePoint(*lastMouse)
        n.correctError(nearest["row"],nearest["col"])
    show()

with open('output.csv', 'w') as file:
    file.write(n.dataAsString())

outputImage=np.zeros((1000,1000,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

cv2.destroyAllWindows()
