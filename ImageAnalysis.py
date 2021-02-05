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

cv2.imshow("window",cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
blankImage=np.zeros((imHeight,imWidth,3), np.uint8)
blankImage[:,:]=(255,0,0)
cv2.imshow("pointsOnly",blankImage)



topLeftIsland=[[11,12],[36,12],[24,17],[26,32]]
topRightIsland=[[533,17],[553,17],[544,24],[543,34]]
bottomLeftIsland=[[14,534],[32,534],[23,538],[24,551]]
bottomRightIsland=[[532,539],[552,539],[543,544],[542,555]]
secondIsland=[[49,11],[67,13],[59,17],[63,29]];

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
    for i in range(x-3,x+3):
        for j in range(y-3,y+3):
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
    c=cv2.circle(im,(int(x),int(y)), 4, col, -1)
    cv2.imshow("window",c)
    c=cv2.circle(blankImage,(int(x),int(y)), 4, col, -1)
    cv2.imshow("pointsOnly",c)

for i in [topLeftIsland,topRightIsland,bottomLeftIsland,bottomRightIsland,secondIsland]:
    for j in i:
        drawPoint(j[0],j[1],(255,0,0))


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
    if event == cv2.EVENT_LBUTTONDOWN:
        print( str(x)+","+str(y))
        drawPoint(x,y,(255,0,0))

        if(len(topLeftIsland)<4):
            topLeftIsland.append([x,y])
        elif(len(secondIsland)<4):
            secondIsland.append([x,y])
        elif(len(topRightIsland)<4):
            topRightIsland.append((x,y))
        elif(len(bottomLeftIsland)<4):
            bottomLeftIsland.append((x,y))
        elif(len(bottomRightIsland)<4):
            bottomRightIsland.append((x,y))
        else:
            m, b =calcBestFit([topLeftIsland[CENTER], topRightIsland[CENTER], secondIsland[CENTER]])#Get the equation y=mx+b for the top row

            islandHSpacing=dist(topLeftIsland[CENTER], secondIsland[CENTER])#Estimate the horizontal spacing using the distance between the centers of the first and second islands
            islandHCount=round(dist(topLeftIsland[CENTER], topRightIsland[CENTER])/islandHSpacing)#Using the previous spacing estimation, determine the number of islands in the top row
            islandHSpacing=round(dist(topLeftIsland[CENTER], topRightIsland[CENTER])/islandHCount)#Now that the number of islands is known, recalculate the spacing using the topleft and topright islands

            #islandTopLeftOffSet=[topLeftIsland[CENTER][0]-topLeftIsland[2]]

            for i in range(islandHCount):
                drawPoint(topLeftIsland[2][0]+i*islandHSpacing,m*(i*islandHSpacing)+b, (255,0,0))








cv2.setMouseCallback('window', mouse_click)

cv2.waitKey(0)
# close all the opened windows.
cv2.destroyAllWindows()
