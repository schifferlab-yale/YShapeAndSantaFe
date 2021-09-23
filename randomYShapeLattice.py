
import random
import cv2

def randomIsland():
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
                    elif(x==width-1):
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


if __name__=="__main__":
    lattice=randomLattice(50,50,allow3InOut=False)
    #img,stats=y_shaped_lattice_analysis.analyze(lattice)
    #print(stats)

    #cv2.imshow("window",img)
    #cv2.waitKey(0)

    with open('randomLattice.csv', 'w') as file:

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