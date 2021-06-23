import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from random import randint
import argparse
import csv

#constants
WHITE=(255,255,255)
BLACK=(0,0,0)
GREEN=(0,255,0)
BLUE=(255,0,0)
RED=(0,0,255)
GREY=(127,127,127)


TOPLEFT=0
TOPRIGHT=1
MIDDLE=2
BOTTOM=3



def analyze(file):

    stats={
        "numRedRings":0,
        "numBlueRings":0,
        "numIslands":0
    }

    #get offset
    if(file[0][0]=="first row offset"):
        firstRowOffset=True
    elif(file[0][0]=="second row offset"):
        firstRowOffset=False
    else:
        raise Exception("no offset data")

    #convert file to data array of ints
    data=[]
    for row in range(2,len(file)):
        data.append([])
        for col in range(len(file[row])):
            if(file[row][col] !=""):
                data[len(data)-1].append(int(file[row][col]))

    #group rows into arrays of 4 to represent each island
    newData=[]
    for row in data:
        newData.append([])
        newData[len(newData)-1].append([])
        i=0
        for char in row:
            lastRow=newData[len(newData)-1]
            lastRow[len(lastRow)-1].append(char)
            i+=1
            if(i==4):
                lastRow.append([])
                i=0
        lastRow=newData[len(newData)-1]
    data=newData

    #
    imageWidth=1000
    padding=50

    outputImage=np.zeros((imageWidth,imageWidth,3), np.uint8)
    outputImage[:,:]=(150,150,150)

    spacingX=(imageWidth-2*padding)/(len(data[0])-1)
    spacingY=(imageWidth-2*padding)/(len(data)-1)

    armLength=min(spacingX,spacingY)/2.5
    offsets=[
        np.array([armLength*math.cos(-5*math.pi/6),armLength*math.sin(-5*math.pi/6)]),
        np.array([armLength*math.cos(-math.pi/6),armLength*math.sin(-math.pi/6)]),
        np.array([0,0]),
        np.array([armLength*math.cos(math.pi/2),armLength*math.sin(math.pi/2)])]

    ys=np.linspace(padding,imageWidth-padding,len(data)-1)
    for (row,y) in enumerate(ys):
        xs=np.linspace(padding,len(data[row])*spacingX,len(data[row]))
        for (col,x) in enumerate(xs):
            island=data[row][col]
            stats["numIslands"]+=1

            #offset if in an offset row
            thisRowOffset=False
            if((row%2==0 and firstRowOffset) or (row%2==1 and not firstRowOffset)):
                thisRowOffset=True
                x+=spacingX/2
            xy=np.array([x,y])


            #draw three points
            for (i,point) in enumerate(island):
                if(point==1):
                    color=WHITE
                elif(point==-1):
                    color=BLACK
                else:
                    color=GREEN
                thisxy=xy+offsets[i]
                #cv2.circle(outputImage,(int(thisxy[0]),int(thisxy[1])), int(spacingX/7), color, -1)



                node=np.array([int(thisxy[0]),int(thisxy[1])])
                center=np.array([int(xy[0]),int(xy[1])])

                nearCenter=tuple(((4*center+node)/5).astype(int))
                node=tuple(node)
                #change arrow direction
                if(color==WHITE):
                    point1=node
                    point2=nearCenter
                else:
                    point2=node
                    point1=nearCenter

                if(color==GREEN):
                    cv2.line(outputImage,point1,point2,color,2)
                else:
                    cv2.arrowedLine(outputImage,point1,point2,color,2,tipLength=spacingX/400)

            #check if this island is the top left of a loop
            if row<len(data)-2 and col<len(data[row])-2:
                right=data[row][col+1]

                if(thisRowOffset):
                    below=data[row+1][col+1]
                else:
                    below=data[row+1][col]

                ringxy=xy+np.array([spacingX/2,spacingY/3])

                color=None
                if(island[TOPRIGHT]==1 and right[TOPLEFT]==-1 and right[BOTTOM]==1 and below[TOPRIGHT]==-1 and below[TOPLEFT]==1 and island[BOTTOM]==-1):
                    color=RED
                    stats["numRedRings"]+=1
                elif(island[TOPRIGHT]==-1 and right[TOPLEFT]==1 and right[BOTTOM]==-1 and below[TOPRIGHT]==1 and below[TOPLEFT]==-1 and island[BOTTOM]==1):
                    color=BLUE
                    stats["numBlueRings"]+=1
                if(color is not None):
                    cv2.circle(outputImage,(int(ringxy[0]),int(ringxy[1])), int(spacingX/4), color, 2)
    return outputImage,stats


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Y shaped lattice csv reader')
    parser.add_argument("file", type=str, help="Path to csv file")
    args=parser.parse_args()

    try:

        file=open(args.file,newline="\n")
    except:
        raise Exception("Error with file")
    

    #clean data and split into lines
    file=file.read().replace("\t","")
    file=file.split("\r\n")
    file=[line.split(", ") for line in file]

    outputImage,stats=analyze(file)

    print(stats)

    cv2.imshow("window",outputImage)
    cv2.waitKey(0)
    cv2.imwrite("analysis-output.jpg", np.float32(outputImage));
