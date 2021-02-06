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

TOPLEFT=0
TOPRIGHT=1
CENTER=2
BOTTOM=3

im = cv2.imread(args.image[0])
imWidth, imHeight, imChannels = im.shape





topLeftIsland=[[11,12],[36,12],[24,17],[26,32]]
topRightIsland=[[533,17],[553,17],[544,24],[543,34]]
bottomLeftIsland=[[14,534],[32,534],[23,538],[24,551]]
bottomRightIsland=[[532,539],[552,539],[543,544],[542,555]]
secondIsland=[[49,11],[67,13],[59,17],[63,29]];

topLeft=[24,17]
topRight=[543,24]
bottomLeft=[23,538]
bottomRight=[543,544]
second=[59,17]

def getAverageColor(im):
    avgColor=0;
    width, height, channels = im.shape
    for i in range(width):
        for j in range(height):
            avgColor+=(int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3
    avgColor/=width*height;
    return [avgColor, avgColor, avgColor]
avgColor=getAverageColor(im)

def getColor(x,y):
    avg=0;
    count=0;
    width=2;
    for i in range(x-width,x+width):
        for j in range(y-width,y+width):
            if x<0 or y<0 or x>imWidth or y>imHeight:
                continue
            avg+=im[y][x][0]
            count+=1
    avg/=count;
    return avg


def drawPoint(x,y,col=WHITE):
    x=int(x)
    y=int(y)
    col=0;
    if(getColor(x,y)>avgColor[0]):
        col=WHITE
    else:
        col=BLACK
    c=cv2.circle(,(int(x),int(y)), 4, col, -1)
    cv2.imshow("window",c)
    c=cv2.circle(blankImage,(int(x),int(y)), 4, col, -1)
    cv2.imshow("pointsOnly",c)

for i in [topLeft, topRight, bottomLeft, bottomRight, second]:
    drawPoint(i[0],i[1],(255,0,0))


def calcBestFit(points):
    #https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
    xMean=0;
    yMean=0;
    for point in points:
        xMean+=point[0]
        yMean+=point[1]
    xMean/=len(points)
    yMean/=len(points)

    mNumerator=0;
    mDenominator=0;
    for point in points:
        x, y = point[0], point[1]
        mNumerator+=(x-xMean)*(y-yMean)
        mDenominator+=(x-xMean)*(x-xMean)

    m=mNumerator/mDenominator
    b=yMean - m*xMean

    return (m, b)
def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))

def mouse_click(event, x, y,flags, param):
    global topLeft, topRight, bottomLeft, bottomRight, second
    if event == cv2.EVENT_MOUSEMOVE:
        drawPoint(x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        print( str(x)+","+str(y))
        drawPoint(x,y,(255,0,0))

        if len(topLeft)==0:
            topLeft=[x,y]
        elif(topRight==[]):
            topRight=[x,y]
        elif(bottomLeft==[]):
            bottomLeft=[x,y]
        elif(bottomRight==[]):
            bottomRight=[x,y]
        elif(second==[]):
            second=[x,y]
        else:

            print(1)
            islandHSpacing=dist(topLeft, second)#Estimate the horizontal spacing using the distance between the centers of the first and second islands
            islandHCount=round(dist(topLeft, topRight)/islandHSpacing)+1#Using the previous spacing estimation, determine the number of islands in the top row
            islandHSpacing=round(dist(topLeft, topRight)/(islandHCount-1))#Now that the number of islands is known, recalculate the spacing using the topleft and topright islands
            print(islandHCount)
            #islandTopLeftOffSet=[topLeftIsland[CENTER][0]-topLeftIsland[2]]

            m, b =calcBestFit([topLeft, second, topRight])#Get the equation y=mx+b for the top row

            for i in range(islandHCount+1):
                drawPoint(topLeft[0]+i*islandHSpacing,m*(i*islandHSpacing)+b, (255,0,0))



class Draggable:
    def __init__(self,x,y):
        self.x=x;
        self.y=y;


def showScreen():
    cv2.imshow("window",cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    blankImage=np.zeros((imHeight,imWidth,3), np.uint8)
    blankImage[:,:]=(255,0,0)
    cv2.imshow("pointsOnly",blankImage)

showScreen();







cv2.setMouseCallback('window', mouse_click)

cv2.waitKey(0)
# close all the opened windows.
cv2.destroyAllWindows()
