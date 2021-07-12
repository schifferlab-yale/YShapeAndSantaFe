import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
from nodeNetwork import *

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
parser.add_argument('-r', "--rows",  help="number of rows", type=int, default=10)
parser.add_argument('-c', "--columns",  help="number of columns", type=int, default=10)
parser.add_argument('-s', "--spacing",  help="how dense the islands are packed small=denser (default=0.25)", type=float, default=0.25)
parser.add_argument("-o", "--offset",  help="Set if the first row is shifted to the right, don't set if the second row is shifted to the right",action='store_true', default=False)
parser.add_argument("-t", "--trim", help="Set if the offset row is shorter than the non-offset rows",action="store_true", default=False)
parser.add_argument("-a", "--reference_image", help="image of the height(to help line up the sample points)", type=str)
args=parser.parse_args()


#read image and reference image
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

shiftConstant=args.spacing;
show_ref_image=False
class YShapeNodeNetwork(NodeNetwork):
    def getSamplePointsFromSquare(self,topLeft,topRight,bottomLeft,bottomRight,row=0,col=0):
        #multiplier for how far the sample points are from the edge of the square

        #get center of sides of square
        centerTop=[topLeft[0]+(topRight[0]-topLeft[0])/2, topLeft[1]+(topRight[1]-topLeft[1])/2]
        centerLeft=[topLeft[0]+(bottomLeft[0]-topLeft[0])/2, topLeft[1]+(bottomLeft[1]-topLeft[1])/2]
        centerRight=[topRight[0]+(bottomRight[0]-topRight[0])/2, topRight[1]+(bottomRight[1]-topRight[1])/2]
        centerBottom=[bottomLeft[0]+(bottomRight[0]-bottomLeft[0])/2, bottomLeft[1]+(bottomRight[1]-bottomLeft[1])/2]

        #square width and height
        width=(centerRight[0]-centerLeft[0])
        height=(centerBottom[1]-centerTop[1])

        #sample points are stored as [x,y,color]
        leftSamplePoint=[topLeft[0]+width*shiftConstant, topLeft[1]+height*shiftConstant]
        rightSamplePoint=[topRight[0]-width*shiftConstant, topRight[1]+height*shiftConstant]
        middleSamplePoint=[(centerLeft[0]+centerRight[0])/2,(centerLeft[1]+centerRight[1])/2]
        bottomSamplePoint=[centerBottom[0], centerBottom[1]-height*shiftConstant]

        samplePoints=[leftSamplePoint, rightSamplePoint, middleSamplePoint, bottomSamplePoint]

        if((row%2==0 and args.offset==True) or (row%2==1 and args.offset==False) ):
            for point in samplePoints:
                point[0]+=width/2

            #odd rows are shorter
            if(args.trim and col+1==self.cols-1):
                return [];



        return samplePoints
    def drawData(self, im):
        if not self.dragging:
            samplePoints=self.samplePoints;

            height, width, channels = im.shape

            for (rowI, row) in enumerate(samplePoints):
                for (vertexI, vertex) in enumerate(row):
                    for (pointI, point) in enumerate(vertex):
                        if(point[2]==1):
                            color=WHITE
                        elif(point[2]==0):
                            color=RED
                        else:
                            color=BLACK
                        if(pointI!=2):
                            im=cv2.line(im,(int(point[0]),int(point[1])),(int(vertex[2][0]),int(vertex[2][1])),color,2)
                        im=cv2.circle(im, (int(point[0]),int(point[1])), 3, color, -1)
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        if(pointI==2):
            sum=0;
            for point in samplePoints[rowI][vertexI]:
                sum+=point[2]
            if(sum==0):
                return False
            else:
                return True
        else:
            return False


n=YShapeNodeNetwork(Node(10,10),Node(800,10),Node(30,800),Node(700,700),args.rows+1, args.columns+1,image)
n.pointSampleWidth=3



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
    outputImage[:,:]=(127,127,127)
    n.drawData(outputImage)
    cv2.imshow("output",outputImage)






def mouse_event(event, x, y,flags, param):
    global lastMouse

    if event == cv2.EVENT_RBUTTONDOWN:
        n.splitAtClosestPoint(x,y)
    elif event ==cv2.EVENT_LBUTTONDOWN:
        if(flags==16 or flags==17 or flags==48):
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

show();
cv2.setMouseCallback('window', mouse_event)

print("Enter: Quit and Save")
print("+/-: Increase/decrease island spacing")
print("r/e: Add/remove row")
print("c/x: Add/remove column")
print("o: toggle row offset")
print("t: toggle row trim")
print("q: toggle reference image")

lastMouse=(0,0)

while(True):
    key=cv2.waitKey(0)

    if(key==ord("\r")):
        break;
    elif(key==ord("+")):
        if shiftConstant<0.5:
            shiftConstant+=0.01
        n.setSamplePoints()
    elif(key==ord("-")):
        if shiftConstant>0:
            shiftConstant-=0.01
        n.setSamplePoints()
    elif(key==ord("r")):
        n.addRow()
    elif(key==ord("e")):
        n.removeRow()

    elif(key==ord("c")):
        n.addCol()
    elif(key==ord("x")):
        n.removeCol()
    elif(key==ord("o")):
        args.offset=not args.offset
        n.setSamplePoints()
    elif(key==ord("t")):
        args.trim=not args.trim
        n.setSamplePoints()
    elif(key==ord("q")):
        show_ref_image=not show_ref_image
    elif(key==ord("j")):
        for i in range(10):
            n.jiggleNearestFixedPoint(*lastMouse)
    show()

with open('output.csv', 'w') as file:
    if(args.offset==True):
        file.write("first row offset\n")
    else:
        file.write("second row offset\n")
    file.write("topLeft, topRight, middle, bottom\n")
    file.write(n.dataAsString())

outputImage=np.zeros((1000,1000,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

cv2.destroyAllWindows()
