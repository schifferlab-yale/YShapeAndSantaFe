import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

WHITE=(255,255,255)
BLACK=(0,0,0)

im = cv2.imread(args.image[0])
im = cv2.resize(im, (1000,1000))
imCopy=im.copy()
imWidth, imHeight, imChannels = im.shape



def mouse_click(event, x, y,flags, param):
    global circles, dragging, selectedPoint
    if event == cv2.EVENT_LBUTTONDOWN:
        selectedPoint=getClosestReferencePoint(x,y)
        dragging=True;
    if event == cv2.EVENT_LBUTTONUP:
        dragging=False;
    elif event == cv2.EVENT_MOUSEMOVE:
        if(dragging):
            referencePoints[selectedPoint]=[x,y]
    elif event == cv2.EVENT_RBUTTONDOWN:
        pass
    showScreen();

#topLeft, topRight, bottomLeft, bottomRight, second, arm
referencePoints=[[43,30],[974,37],[41, 962],[103, 34], [42,55]]
selectedPoint=-1;
dragging=False;

def getColor(x,y):
    avg=0;
    count=0;
    width=4;
    x=int(x)
    y=int(y)
    for i in range(x-width,x+width):
        for j in range(y-width,y+width):
            if x<0 or y<0 or x>imWidth-1 or y>imHeight-1:
                continue
            avg+=im[y][x][0]
            count+=1
    if count==0:
        return 0
    avg/=count;

    return avg
def getAverageColor(im):
    avgColor=0;
    width, height, channels = im.shape
    for i in range(width):
        for j in range(height):
            avgColor+=(int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3
    avgColor/=width*height;
    return [avgColor, avgColor, avgColor]
avgColor=getAverageColor(im)

def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))
def getClosestReferencePoint(x,y):
    minDist=100000;
    selected=0;
    for (i,point) in enumerate(referencePoints):
        distance=dist([x,y],point)
        if(distance<minDist):
            selected=i;
            minDist=distance;
    return selected
def addPoint(x,y, im, blank):
    im=cv2.circle(im,(int(x),int(y)),5,(255,0,0),2)
    if getColor(x,y)<avgColor[0]:
        blank=cv2.circle(blank,(int(x),int(y)),5,BLACK,2)
    else:
        blank=cv2.circle(blank,(int(x),int(y)),5,WHITE,2)
def showScreen():
    blankImage=np.zeros((imHeight,imWidth,3), np.uint8)
    blankImage[:,:]=(255,0,0)

    im=imCopy.copy();
    for point in referencePoints:
        im=cv2.circle(im,(point[0],point[1]), 10, (255,0,0,0.1), 3)
    #l=cv2.line(im,(referencePoints[0][0], referencePoints[0][1]), (referencePoints[1][0], referencePoints[1][1]), (0,0,255), 2)
    topLeft=referencePoints[0]
    topRight=referencePoints[1]
    bottomLeft=referencePoints[2]
    second=referencePoints[3]
    arm=referencePoints[4]

    hSpacing=dist(topLeft,second)
    hCount=round(dist(topRight,topLeft)/hSpacing+1);
    hSpacing=dist(topLeft,topRight)/hCount;

    vSpacing=hSpacing*math.sqrt(3)/2
    vCount=round(dist(topLeft, bottomLeft)/vSpacing+1);
    vSpacing=dist(topLeft,topRight)/vCount;

    armDist=dist(topLeft, arm)
    armAngle=math.atan2(topLeft[1]-arm[1],topLeft[0]-arm[0]);

    rowIsOdd=False;
    for (rowX, rowY) in zip(np.linspace(topLeft[0], bottomLeft[0],vCount),np.linspace(topLeft[1], bottomLeft[1],vCount)):
        if(rowIsOdd):
            rowX+=hSpacing/2
        rowIsOdd=not rowIsOdd
        for (colX, colY) in zip(np.linspace(rowX,rowX+hCount*hSpacing, hCount ) ,np.linspace(rowY, rowY+(topRight[1]-topLeft[1]),hCount)):
            #im=cv2.circle(im,(int(colX),int(colY)),5,(0,0,255),2)
            addPoint(colX,colY, im, blankImage)
            addPoint(colX-armDist*math.cos(armAngle),colY-armDist*math.sin(armAngle),im, blankImage)
            addPoint(colX-armDist*math.cos(armAngle+2*math.pi/3),colY-armDist*math.sin(armAngle+2*math.pi/3),im,blankImage)
            addPoint(colX-armDist*math.cos(armAngle+4*math.pi/3),colY-armDist*math.sin(armAngle+4*math.pi/3),im,blankImage)
            #addPoint(colX+armDist*math.cos(armAngle+2*math.pi/3), colY+armDist*math.sin(armAngle+2*math.pi/3),im)
            #im=cv2.circle(im,(int(colX+armDist*math.cos(armAngle)),int(colY+armDist*math.sin(armAngle))),5,(0,255,0),2)
            #im=cv2.circle(im,(int(colX+armDist*math.cos(armAngle+math.pi*2/3)),int(colY+armDist*math.sin(armAngle+math.pi*2/3))),5,(0,255,0),2)
            #im=cv2.circle(im,(int(colX+armDist*math.cos(armAngle+math.pi*4/3)),int(colY+armDist*math.sin(armAngle+math.pi*4/3))),5,(0,255,0),2)





    print(armAngle)
    cv2.imshow("nopic",blankImage)
    cv2.imshow("window",im)






showScreen();
cv2.setMouseCallback('window', mouse_click)

cv2.waitKey(0)
# close all the opened windows.
cv2.destroyAllWindows()
