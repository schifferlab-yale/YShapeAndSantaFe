import random
import cv2
from y_shaped_lattice_analysis_new import getIslandAngle

#Each of the four points in an island
TOPLEFT=0
TOPRIGHT=1
MIDDLE=2
BOTTOM=3


#for indexing the neighbors of an island
TOPLEFTNEIGHBOR=0
TOPRIGHTNEIGHBOR=1
LEFTNEIGHBOR=2
RIGHTNEIGHBOR=3
BOTTOMLEFTNEIGHBOR=4
BOTTOMRIGHTNEIGHBOR=5


def randomIsland(plusChargeOnly=False):
    
    if plusChargeOnly:
        islands=[
            [-1,1,-1,1],
            [1,-1,-1,1],
            [1,1,-1,-1],
        ]
    else:
        islands=[
            [1,-1,1,-1],
            [-1,1,1,-1],
            [-1,-1,1,1],
            [-1,1,-1,1],
            [1,-1,-1,1],
            [1,1,-1,-1],
        ]
    #return islands[0]
    return random.choice(islands)

def randomLattice(width,height,allow3InOut=True):
    lattice=[]

    firstRowOffset=random.choice([True,False])

    lattice.append(["first row offset" if firstRowOffset else "second row offset"])
    lattice.append(["topLeft, topRight, middle, bottom"])
    for y in range(2,height+2):
        newLine=[]

        #offset if in an offset row
        thisRowOffset=False
        if((y%2==0 and firstRowOffset) or (y%2==1 and not firstRowOffset)):
            thisRowOffset=True

        for x in range(width):

            if allow3InOut:
                nextIsland=randomIsland()
            else:
                valid=False

                while not valid:
                    nextIsland=randomIsland()

                    if(len(newLine)<3):
                        valid=True
                    elif(y==2):
                        valid=True
                    elif(x==width):
                        valid=True
                    else:
                        thisNode=nextIsland[0]
                        leftNode=newLine[len(newLine)-3]
                        upNode=None
                        if(thisRowOffset):

                            upNode=lattice[y-1][4*(x)+3]
                        else:
                            upNode=lattice[y-1][4*(x-1)+3]

                        if(leftNode!=thisNode or leftNode!=upNode):
                            valid=True

            newLine+=nextIsland
        lattice.append(newLine)
    
    return lattice

def randomChargeOrderedLattice(width,height):
    lattice=[]

    firstRowOffset=random.choice([True,False])

    

    unfilledCells=[]

    for i in range(height):
        row=[]
        for j in range(width):
            row.append(None)
            unfilledCells.append((i,j))
        lattice.append(row)
    
    fillChargeOrderedLattice(lattice,unfilledCells,firstRowOffset)
    print(lattice)


    for i in range(len(lattice)):
        rowConcatenated=[]
        for cell in lattice[i]:
            rowConcatenated+=cell
        lattice[i]=rowConcatenated

        


    lattice.insert(0,["topLeft, topRight, middle, bottom"])
    lattice.insert(0,["first row offset" if firstRowOffset else "second row offset"])

    
    print(lattice)
    return lattice


def fillChargeOrderedLattice(lattice,unfilledCells,firstRowOffset):
    if len(unfilledCells)==0:
        return lattice
    random.shuffle(unfilledCells)
    for cellAddress in unfilledCells:
        possibleMoments=[
            [-1,1,-1,1],
            [1,-1,-1,1],
            [1,1,-1,-1],
        ]
        random.shuffle(possibleMoments)

        for moment in possibleMoments:
            try:
                lattice[cellAddress[0]][cellAddress[1]]=moment
            except Exception:
                print(lattice)
                raise Exception
            if isValidChargeOrderedCell(lattice,cellAddress,firstRowOffset):
                
                newLattice=fillChargeOrderedLattice(lattice,[cell for cell in unfilledCells if cell!=cellAddress],firstRowOffset)
                if newLattice is None:
                    continue
                else:
                    unfilledCells.remove(cellAddress)  
                    return newLattice
            else:
                lattice[cellAddress[0]][cellAddress[1]]=None
    return None

def isValidChargeOrderedCell(lattice,cellAddress,firstRowOffset):
    cellRow=cellAddress[0]
    cellCol=cellAddress[1]

    offset=(cellRow%2==0 and firstRowOffset) or (cellRow%2==1 and not firstRowOffset)

    rightCell=getCellOrNone(lattice,cellRow,cellCol+1)
    leftCell=getCellOrNone(lattice,cellRow,cellCol-1)
    if offset:
        topLeftCell=getCellOrNone(lattice,cellRow-1,cellCol)
        topRightCell=getCellOrNone(lattice,cellRow-1,cellCol+1)
        bottomLeftCell=getCellOrNone(lattice,cellRow+1,cellCol)
        bottomRightCell=getCellOrNone(lattice,cellRow+1,cellCol+1)
    else:
        topLeftCell=getCellOrNone(lattice,cellRow-1,cellCol-1)
        topRightCell=getCellOrNone(lattice,cellRow-1,cellCol)
        bottomLeftCell=getCellOrNone(lattice,cellRow+1,cellCol-1)
        bottomRightCell=getCellOrNone(lattice,cellRow+1,cellCol)
    
    angle=getIslandAngle(lattice[cellRow][cellCol])
    while angle<0: angle+=360

    topLeftAngle=getIslandAngle(topLeftCell)
    topRightAngle=getIslandAngle(topRightCell)
    leftAngle=getIslandAngle(leftCell)
    rightAngle=getIslandAngle(rightCell)
    bottomLeftAngle=getIslandAngle(bottomLeftCell)
    bottomRightAngle=getIslandAngle(bottomRightCell)

    rightCharge=(angle==330)-(angle==150)+(topRightAngle==90)-(topRightAngle==270)+(rightAngle==210)-(rightAngle==30)
    leftCharge=(angle==210)-(angle==30)+(topLeftAngle==90)-(topLeftAngle==270)+(leftAngle==330)-(leftAngle==150)
    bottomCharge=(angle==90)-(angle==270)+(bottomLeftAngle==330)-(bottomLeftAngle==150)+(bottomRightAngle==210)-(bottomRightAngle==30)

    if topRightAngle is not None and rightAngle is not None and rightCharge!=1:
        return False
    if topLeftAngle is not None and leftAngle is not None and leftCharge!=1:
        return False
    if bottomLeftAngle is not None and bottomRightAngle is not None and bottomCharge!=1:
        return False
    return True


    """if angle==30:
        if getIslandAngle(rightCell)==150 and getIslandAngle(bottomRightCell)==270:
            return False
    elif angle==90:
        if getIslandAngle(bottomRightCell)==210 and getIslandAngle(bottomLeftCell)==330:
            return False
    elif angle==150:
        if getIslandAngle(bottomLeftCell)==270 and getIslandAngle(leftCell)==30:
            return False
    elif angle==210:
        if getIslandAngle(leftCell)==330 and getIslandAngle(topLeftCell)==90:
            return False
    elif angle==270:
        if getIslandAngle(topLeftCell)==30 and getIslandAngle(topRightCell)==150:
            return False
    elif angle==330:
        
        if getIslandAngle(topRightCell)==90 and getIslandAngle(rightCell)==210:
            return False
    else:
        raise Exception(f"Bad angle: {angle}")
    return True"""



def getCellOrNone(lattice,row,col):
    if row<0 or row>=len(lattice):
        return None
    if col<0 or col>=len(lattice[row]):
        return None
    return lattice[row][col]



if __name__=="__main__":
    
    #img,stats=y_shaped_lattice_analysis.analyze(lattice)
    #print(stats)

    #cv2.imshow("window",img)
    #cv2.waitKey(0)

    for i in range(1):
        lattice=randomChargeOrderedLattice(5,5)
        with open(f'randomLatticeChargeOrdered_{i}.csv', 'w') as file:
            string= "\n\r".join([el[0] for el in[[", ".join([str(el) for el in row])] for row in lattice]])

            file.write(string)

    """overall={"numRedRings":0,"numBlueRings":0,"numIslands":0}
    numRuns=100
    for i in range(numRuns):
        lattice=randomLattice(40,40,allow3InOut=False)
        img,stats=y_shaped_lattice_analysis.analyze(lattice)
        overall["numBlueRings"]+=stats["numBlueRings"]
        overall["numRedRings"]+=stats["numRedRings"]
        overall["numIslands"]+=stats["numIslands"]

    overall["numRedRings"]/=numRuns
    overall["numBlueRings"]/=numRuns
    overall["numIslands"]/=numRuns
    print(overall)"""