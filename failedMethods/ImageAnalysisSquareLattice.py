import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

#Get the image and make sure it exists
image=[];
try:
    image = cv2.imread(args.image[0])
    image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")




#CONSTANTS
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(255,0,0)
RED=(0,0,255)
sampleWidth=10;


def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))



class PointGrid:
    def __init__(self,referencePoints,im):
        self.referencePoints=referencePoints;
        self.grid=self.genGrid(referencePoints,im)
    def draw(self,im):
        for point in self.grid:
            im=cv2.circle(im, (int(point[0]),int(point[1])), 2, RED, -1)
    def genGrid(self,referencePoints):
        pass

class SquarePointGrid(PointGrid):
    def genGrid(self,referencePoints,im):
        topLeft=referencePoints[0]
        topRight=referencePoints[1]
        bottomLeft=referencePoints[2]
        second=referencePoints[3]

        spacing=dist(topLeft,second)
        hCount=round(dist(topRight,topLeft)/spacing+1);
        hSpacing=dist(topLeft,topRight)/hCount;

        vCount=round(dist(topLeft, bottomLeft)/spacing+1);#estimate number of vertical islansd
        vSpacing=dist(topLeft,topRight)/vCount;#get a better value for vspacing now that vcount is known

        grid=[]
        for (rowX, rowY) in zip(np.linspace(topLeft[0], bottomLeft[0],vCount),np.linspace(topLeft[1], bottomLeft[1],vCount)):
            for (colX, colY) in zip(np.linspace(rowX,rowX+hCount*hSpacing, hCount ) ,np.linspace(rowY, rowY+(topRight[1]-topLeft[1]),hCount)):
                grid.append([colX,colY])

        return grid




class ImageAnalyzer:
    def __init__(self, im, numReferencePoints, referenceLabels, referencePoints=[]):
        assert(numReferencePoints==len(referenceLabels))
        self.numReferencePoints=numReferencePoints
        self.referenceLabels=referenceLabels
        self.referencePoints=[]
        self.rawIm=im.copy();
        self.imWidth, self.imHeight, self.imChannels = im.shape
        self.sampleWidth=0;
        self.dragging=False
        self.selectedPoint=-1;

        self.grid=None;

        self.avgColor=self.getAverageColor()
        self.stdColor=self.getStdColor()

        self.thresholdImage=self.getThresholdImage(1.25);
        cv2.imshow("thresh", self.thresholdImage)

        print(self.avgColor)
        print(self.stdColor)
        self.windowName="window"


        self.showImage()
        #cv2.setMouseCallback(self.windowName, self.mouse_event)


        if(len(referencePoints)<numReferencePoints):
            print("Please click on "+referenceLabels[0])

    def getAverageColor(self):
        avgColor=0;
        im=self.rawIm
        for i in range(self.imWidth):
            for j in range(self.imHeight):
                avgColor+=(int(im[i][j][0]))
        avgColor/=self.imWidth*self.imHeight;
        return avgColor
    def getStdColor(self):
        stdColor=0;
        im=self.rawIm
        for i in range(self.imWidth):
            for j in range(self.imHeight):
                stdColor+=abs((int(im[i][j][0]))-self.avgColor);
        stdColor/=self.imWidth*self.imHeight;
        return stdColor


    def getColor(self,x,y):
        avg=0;
        count=0;#number of pixels checked
        width=self.sampleWidth;#distance away from center pixel to sample
        x=int(x)
        y=int(y)
        im=self.rawIm
        #loop through rows and columns
        for i in range(x-width,x+width+1):
            for j in range(y-width,y+width+1):
                #make sure pixel is not offscreen
                if i<0 or j<0 or i>self.imWidth-1 or j>self.imHeight-1:
                    continue
                #add to average
                avg+=(im[j][i][0])#+im[j][i][1]+im[j][i][2])/3
                count+=1

        #prevent divide by 0 error
        if count==0:
            return 0

        #return avg color
        avg/=count;
        return avg
    def getThresholdImage(self, stdScale):
        im=self.rawIm.copy();
        for row in range(len(self.rawIm)):
            for col in range(len(self.rawIm[row])):
                color=self.getColor(col,row)
                if(color>self.avgColor+self.stdColor*stdScale):
                    im[row][col]=[255,255,255]
                elif(color<self.avgColor-self.stdColor*stdScale):
                    im[row][col]=[0,0,0]
                else:
                    im[row][col]=[127,127,127]
        return im

    def getClosestReferencePoint(self,x,y):
        minDist=100000;
        selected=0;
        for (i,point) in enumerate(self.referencePoints):
            distance=dist([x,y],point)
            if(distance<minDist):
                selected=i;
                minDist=distance;
        return selected


    def mouse_event(self,event, x, y,flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if(len(self.referencePoints)<self.numReferencePoints):
                self.referencePoints.append([x,y])
                if(len(self.referencePoints)<self.numReferencePoints):
                    print("Please click on "+self.referenceLabels[len(self.referencePoints)])
                else:
                    self.grid=SquarePointGrid(self.referencePoints)
            else:
                self.dragging=True;
                self.selectedPoint=self.getClosestReferencePoint(x,y)
        #Release dragging reference point
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging=False;
            self.grid=SquarePointGrid(self.referencePoints,self.rawIm)


        #Update reference point if dragging
        elif event == cv2.EVENT_MOUSEMOVE:
            if(self.dragging):
                self.referencePoints[self.selectedPoint]=[x,y]
        self.showImage();


    def showImage(self):
        im=self.rawIm.copy();
        for (label,point) in zip(self.referenceLabels, self.referencePoints):
            im=cv2.circle(im,(point[0],point[1]), sampleWidth, BLUE, 5)
            if self.dragging:
                im=cv2.putText(im, label, (point[0],point[1]+20),cv2.FONT_HERSHEY_SIMPLEX ,0.4,(0,0,255),1,cv2.LINE_AA)

        if self.grid:
            self.grid=SquarePointGrid(self.referencePoints,im)
            self.grid.draw(im)


        cv2.imshow(self.windowName,self.rawIm)






ImageAnalyzer=ImageAnalyzer(image, 4, ["Top Left", "Top Right", "Bottom Left", "Second"])

cv2.waitKey(0)


# close all the opened windows.
cv2.destroyAllWindows()
