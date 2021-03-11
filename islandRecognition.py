import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('height', metavar='image', type=str, nargs='+',help='Path of image')
parser.add_argument('phase', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()


IMSIZE=800

try:
    heightIm = cv2.imread(args.height[0])
    heightIm = cv2.resize(heightIm, (IMSIZE, IMSIZE))
except:
    raise Exception("File not found")
try:
    phaseIm = cv2.imread(args.phase[0])
    phaseIm = cv2.resize(phaseIm, (IMSIZE, IMSIZE))
except:
    raise Exception("File not found")


heightIm= cv2.erode(heightIm,(3,3), iterations=3)


#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)





class Drawer():
    def __init__(self,im,phaseIm):
        self.originalHeightIm=im.copy()
        self.modifiedHeightIm=im.copy()
        self.phaseIm=phaseIm.copy()

        self.HEIGHT_THRESHOLD=127;

        self.drawing=None;
        self.lastMousePos=(0,0)

        self.centers=[]

        self.pointSampleWidth=1

        self.centerClickLoc=(0,0)
        self.centerClicking=False

        self.radius=22
        self.angle=-math.pi/2

        cv2.imshow("phaseIm", self.phaseIm)
        cv2.imshow("heightIm",self.modifiedHeightIm)

        #cv2.setMouseCallback('draw', self.mouse_event)
        cv2.setMouseCallback('heightIm', self.heightImMouseEvent)
        cv2.setMouseCallback('phaseIm', self.phaseImMouseEvent)



        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def heightImMouseEvent(self,event, x, y,flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drawing=WHITE
        elif event ==cv2.EVENT_LBUTTONDOWN:
            self.drawing=BLACK
        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing is not None:
                #cv2.circle(self.modified,(x,y),4,self.drawing,-1)
                cv2.line(self.modifiedHeightIm,(x,y),self.lastMousePos,self.drawing,4)
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing=None
        elif event==cv2.EVENT_RBUTTONUP:
            self.drawing=None


        self.drawHeightIm()
        self.drawPhaseIm()
        self.lastMousePos=(x,y)

    def phaseImMouseEvent(self,event, x, y,flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            pass
        elif event ==cv2.EVENT_LBUTTONDOWN:
            self.centerClickLoc=(x,y)
            self.centerClicking=True
        elif event==cv2.EVENT_MOUSEMOVE:
            pass
        elif event==cv2.EVENT_LBUTTONUP:
            if self.centerClicking:
                self.centerClicking=False
                self.radius=math.sqrt( math.pow(self.centerClickLoc[0]-x,2) +math.pow(self.centerClickLoc[1]-y,2))
                self.angle=-math.atan2(y-self.centerClickLoc[1],x-self.centerClickLoc[0])
        elif event==cv2.EVENT_RBUTTONUP:
            pass


        self.drawPhaseIm()

    def drawPhaseIm(self):
        im=self.phaseIm.copy()
        for c in self.centers:
            samplePoints=[c]
            samplePoints.append( (int(c[0]+self.radius*math.cos(self.angle)), int(c[1]-self.radius*math.sin(self.angle))) )
            samplePoints.append( (int(c[0]+self.radius*math.cos(self.angle+2*math.pi/3)), int(c[1]-self.radius*math.sin(self.angle+2*math.pi/3))) )
            samplePoints.append( (int(c[0]+self.radius*math.cos(self.angle+4*math.pi/3)), int(c[1]-self.radius*math.sin(self.angle+4*math.pi/3))) )

            for p in samplePoints:
                if(self.sampleImageColor(*p)==1):
                    im=cv2.circle(im,p,4,BLUE,-1)
                else:
                    im=cv2.circle(im,p,4,RED,-1)
        im=cv2.circle(im,self.lastMousePos,5,GREEN,-1)
        cv2.imshow("phaseIm", im)

    def sampleImageColor(self,x,y):
        avg=0;
        count=0;#number of pixels checked
        width=self.pointSampleWidth;#distance away from center pixel to sample
        x=int(x)
        y=int(y)

        im=self.phaseIm

        imHeight, imWidth, channels=im.shape

        #loop through rows and columns
        for i in range(x-width,x+width):
            for j in range(y-width,y+width):
                #make sure pixel is not offscreen
                if i<0 or j<0 or i>imWidth-1 or j>imHeight-1:
                    continue
                #add to average
                avg+=im[j][i][0]#(im[j][i][0]+im[j][i][1]+im[j][i][2])/3
                count+=1

        #prevent divide by 0 error
        if count==0:
            return 0

        #return avg color
        avg/=count;
        if(avg>127):
            return 1
        return -1

    def drawHeightIm(self):
        ret,BWHeightIm=cv2.threshold(self.modifiedHeightIm,self.HEIGHT_THRESHOLD,255,cv2.THRESH_BINARY)

        imgray = cv2.cvtColor(BWHeightIm,cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(imgray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(self.modifiedHeightIm.copy(), contours, -1, RED, 1)

        self.centers=[]
        for c in contours:
        	# compute the center of the contour
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv2.circle(image, (cX, cY), 4, BLUE, -1)
                self.centers.append((cX,cY))

            except:
                pass

        cv2.imshow("heightIm",image)

d=Drawer(heightIm,phaseIm)
