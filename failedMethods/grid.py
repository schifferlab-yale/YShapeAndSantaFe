import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint



try:
    image = cv2.imread("squareLattice.jpeg")
    image = cv2.resize(image, (1000,1000))
except:
    raise Exception("File not found")


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
        self.parent=None
        self.selectedRow=-1
        self.selectedCol=-1
        self.selected=False

        self.subgrids=[]

        self.grid=[];
        self.calcGrid()

    def calcGrid(self):
        self.grid=[]
        topLeft=self.topLeft
        topRight=self.topRight
        bottomLeft=self.bottomLeft
        bottomRight=self.bottomRight
        rows=self.rows
        cols=self.cols
        hSpacing=dist(topLeft,topRight)/cols;
        for (rowStartX, rowStartY, rowEndX, rowEndY) in zip(np.linspace(topLeft[0], bottomLeft[0],rows), np.linspace(topLeft[1], bottomLeft[1], rows), np.linspace(topRight[0],bottomRight[0],rows), np.linspace(topRight[1],bottomRight[1],rows)):
            self.grid.append([])
            for (colX, colY) in zip(np.linspace(rowStartX,rowEndX, cols ), np.linspace(rowStartY,rowEndY, cols )):
                self.grid[len(self.grid)-1].append([colX,colY])

    def draw(self,im):
        if(len(self.subgrids)==0):
            im=cv2.line(im, (int(self.topLeft[0]),int(self.topLeft[1])), (int(self.topRight[0]),int(self.topRight[1])), RED, 2)
            im=cv2.line(im, (int(self.topLeft[0]),int(self.topLeft[1])), (int(self.bottomLeft[0]),int(self.bottomLeft[1])), RED, 2)
            im=cv2.line(im, (int(self.bottomRight[0]),int(self.bottomRight[1])), (int(self.topRight[0]),int(self.topRight[1])), RED, 2)
            im=cv2.line(im, (int(self.bottomRight[0]),int(self.bottomRight[1])), (int(self.bottomLeft[0]),int(self.bottomLeft[1])), RED, 2)
            #for row in self.grid:
                #for point in row:
                    #im=cv2.circle(im,(int(point[0]),int(point[1])),5,self.color,-1)

            for row in self.grid:
                im=cv2.line(im,(int(row[0][0]),int(row[0][1])),(int(row[len(row)-1][0]),int(row[len(row)-1][1])),RED,1)
            for i in range(len(self.grid[0])):
                im=cv2.line(im, (int(self.grid[0][i][0]),int(self.grid[0][i][1])), (int(self.grid[len(self.grid)-1][i][0]),int(self.grid[len(self.grid)-1][i][1])), RED, 1)

        else:
            for grid in self.subgrids:
                grid.draw(im)

        #if self.selected:
        #    cv2.circle(im,(int(self.getCenter()[0]),int(self.getCenter()[1])),20,BLACK,3)
        #else:
        #    cv2.circle(im,(int(self.getCenter()[0]),int(self.getCenter()[1])),20,RED,3)

    def getBaseGrids(self):
        if(len(self.subgrids)==0):
            return [self]
        else:
            subgrids=[]
            for grid in self.subgrids:
                subgrids=subgrids+grid.getBaseGrids();

        return subgrids
    def getAllGrids(self):
        subgrids=[]
        for grid in self.subgrids:
            subgrids=subgrids+grid.getAllGrids();
        subgrids.append(self)
        return subgrids

    def splitPoint(self,x,y):
        gridToSplit=None;
        minDist=100000;
        splitRow=0;
        splitCol=0;
        for grid in self.getBaseGrids():
            for (rowIndex, row) in enumerate(grid.grid):
                for(colIndex, point) in enumerate(row):
                    distance=dist(point, [x,y])
                    if distance<minDist:
                        minDist= distance
                        gridToSplit=grid
                        splitRow=rowIndex
                        splitCol=colIndex

        gridToSplit.split(splitRow,splitCol)

    def getCenter(self):
        if(len(self.subgrids)==0):
            centerX=(self.topLeft[0]+self.topRight[0]+self.bottomLeft[0]+self.bottomRight[0])/4
            centerY=(self.topLeft[1]+self.topRight[1]+self.bottomLeft[1]+self.bottomRight[1])/4
            return [centerX,centerY]
        else:
            return self.subgrids[0].bottomRight

    def selectGrid(self,x,y):
        selectedGrid=None;
        minDist=100000;
        for grid in self.getAllGrids():
            grid.selected=False
            distance=dist(grid.getCenter(), [x,y])
            if(distance<minDist and len(grid.subgrids)==4 and len(grid.subgrids[0].subgrids)==0):
                selectedGrid=grid
                minDist=distance
        if(selectedGrid!=None):
            selectedGrid.selected=True

    def getSelectedGrid(self):
        for grid in self.getAllGrids():
            if grid.selected==True:
                return grid

    def deselectAll(self):
        for grid in self.getAllGrids():
            grid.selected=False;

    def setTopLeft(self,x,y):
        self.topLeft=[x,y]
        if(len(self.subgrids)>0):
            self.subgrids[0].setTopLeft(x,y)
        self.calcGrid()
    def setTopRight(self,x,y):
        self.topRight=[x,y]
        if(len(self.subgrids)>0):
            self.subgrids[1].setTopRight(x,y)
        self.calcGrid()
    def setBottomLeft(self,x,y):
        self.bottomLeft=[x,y]
        if(len(self.subgrids)>0):
            self.subgrids[2].setBottomLeft(x,y)
        self.calcGrid()
    def setBottomRight(self,x,y):
        self.bottomRight=[x,y]
        if(len(self.subgrids)>0):

            self.subgrids[3].setBottomRight(x,y)
        self.calcGrid()

    def setCenter(self,x,y):
        grids=self.subgrids
        grids[0].setBottomRight(x,y)
        grids[1].setBottomLeft(x,y)
        grids[2].setTopRight(x,y)
        grids[3].setTopLeft(x,y)
        for grid in grids:
            grid.calcGrid()

    def getCorners(self):
        return [self.topLeft, self.topRight, self.bottomLeft, self.bottomRight]

    def setCornerByIndex(self,index,x,y):
        if index==0:
            self.setTopLeft(x,y)
        elif index==1:
            self.setTopRight(x,y)
        elif index==2:
            self.setBottomLeft(x,y)
        elif index==3:
            self.setBottomRight(x,y)




    def split(self,row,col):
        #topLeft, topRight, bottomLeft, bottomRight
        grid1=Grid(self.topLeft, self.grid[0][col], self.grid[row][0],self.grid[row][col],  row+1, col+1)
        grid2=Grid(self.grid[0][col], self.grid[0][self.cols-1], self.grid[row][col], self.grid[row][self.cols-1], row+1, self.cols-col)
        grid3=Grid(self.grid[row][0],self.grid[row][col],self.grid[self.rows-1][0],self.grid[self.rows-1][col], self.rows-row, col+1)
        grid4=Grid(self.grid[row][col], self.grid[row][self.cols-1],self.grid[self.rows-1][col],self.grid[self.rows-1][self.cols-1],self.rows-row,self.cols-col)

        grids=[grid1,grid2,grid3,grid4]

        grids= [grid for grid in grids if grid.rows != 1 and grid.cols!=1]
        if(len(grids)<4):
            return;

        for grid in grids:
            grid.color=(randint(0,255),randint(0,255),randint(0,255),randint(0,255))
            grid.parent=self;

        self.subgrids=grids




g=Grid([10,10],[900,30],[40,900],[600,600],25,25)
print(g.getBaseGrids())

def show():
    imWidth=1000;
    imHeight=1000;
    #outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    #outputImage[:,:]=(255,255,255)
    outputImage=image.copy()

    g.draw(outputImage)



    cv2.imshow("window",outputImage)
dragging=False
adjustingCorner=-1
def mouse_event(event, x, y,flags, param):
    global dragging, adjustingCorner
    if event == cv2.EVENT_LBUTTONDOWN:
        for (i, corner) in enumerate(g.getCorners()):
            if(dist(corner,[x,y])<10):
                adjustingCorner=i
        if(adjustingCorner==-1):
            g.selectGrid(x,y)
            dragging=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if(adjustingCorner>=0):
            g.setCornerByIndex(adjustingCorner,x,y)
            g.calcGrid()
        elif(dragging):
            g.getSelectedGrid().setCenter(x,y)
    elif event==cv2.EVENT_LBUTTONUP:
        g.deselectAll()
        dragging=False
        adjustingCorner=-1;
    elif event == cv2.EVENT_RBUTTONDOWN:
        g.splitPoint(x,y)

    show()

show();
cv2.setMouseCallback('window', mouse_event)
cv2.waitKey(0)

cv2.destroyAllWindows()
