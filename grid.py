import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint



WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(255,0,0)
RED=(0,0,255)


def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))

class Grid:
    def __init__(self,topLeft,topRight,bottomLeft,bottomRight,rows,cols):
        self.topLeft=topLeft;
        self.topRight=topRight;
        self.bottomLeft=bottomLeft;
        self.bottomRight=bottomRight;
        self.rows=rows;
        self.cols=cols;
        self.color=RED

        self.subgrids=[]

        self.grid=[];
        hSpacing=dist(topLeft,topRight)/cols;
        for (rowStartX, rowStartY, rowEndX, rowEndY) in zip(np.linspace(topLeft[0], bottomLeft[0],rows), np.linspace(topLeft[1], bottomLeft[1], rows), np.linspace(topRight[0],bottomRight[0],rows), np.linspace(topRight[1],bottomRight[1],rows)):
            self.grid.append([])
            for (colX, colY) in zip(np.linspace(rowStartX,rowEndX, cols ), np.linspace(rowStartY,rowEndY, cols )):
                self.grid[len(self.grid)-1].append([colX,colY])

    def draw(self,im):
        if(len(self.subgrids)==0):
            im=cv2.line(im, (int(self.topLeft[0]),int(self.topLeft[1])), (int(self.topRight[0]),int(self.topRight[1])), BLACK, 2)
            im=cv2.line(im, (int(self.topLeft[0]),int(self.topLeft[1])), (int(self.bottomLeft[0]),int(self.bottomLeft[1])), BLACK, 2)
            im=cv2.line(im, (int(self.bottomRight[0]),int(self.bottomRight[1])), (int(self.topRight[0]),int(self.topRight[1])), BLACK, 2)
            im=cv2.line(im, (int(self.bottomRight[0]),int(self.bottomRight[1])), (int(self.bottomLeft[0]),int(self.bottomLeft[1])), BLACK, 2)
            for row in self.grid:
                for point in row:
                    im=cv2.circle(im,(int(point[0]),int(point[1])),5,self.color,-1)
        else:
            for grid in self.subgrids:
                grid.draw(im)

    def getClosestPoint(self,x,y):
        if(len(self.subgrids)==0):
            minDist=10000;
            x=0;
            y=0
            #for (y,row) in enumerate(self.grid):
                #for point in row:############################


    def split(self,row,col):
        #topLeft, topRight, bottomLeft, bottomRight
        grid1=Grid(self.topLeft, self.grid[0][col], self.grid[row][0],self.grid[row][col], row+1, col+1)
        grid2=Grid(self.grid[0][col], self.grid[0][self.cols-1], self.grid[row][col], self.grid[row][self.cols-1], row+1, self.cols-row)
        grid3=Grid(self.grid[row][0],self.grid[row][col],self.grid[self.rows-1][0],self.grid[self.rows-1][col],self.rows-col,col+1)
        grid4=Grid(self.grid[row][col], self.grid[row][self.cols-1],self.grid[self.rows-1][col],self.grid[self.rows-1][self.cols-1],self.rows-col,self.cols-row)


        self.subgrids=[grid1,grid2,grid3,grid4]
        for grid in self.subgrids:
            grid.color=(randint(0,255),randint(0,255),randint(0,255),randint(0,255))



g=Grid([10,10],[900,30],[40,900],[600,600],20,15)
g.split(9,9)
g.subgrids[0].split(3,3)

def show():
    imWidth=1000;
    imHeight=1000;
    outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    outputImage[:,:]=(255,255,255)

    g.draw(outputImage)



    cv2.imshow("window",outputImage)

def mouse_event(event, x, y,flags, param):
    show()

show();
cv2.setMouseCallback('window', mouse_event)
cv2.waitKey(0)

cv2.destroyAllWindows()
