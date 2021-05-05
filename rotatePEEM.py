import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math




def coordsToArray(row,col,islands):
    assert(len(row)==len(col) and len(col)==len(islands))
    row=np.array(row)
    col=np.array(col)

    minRow=np.min(row)
    maxRow=np.max(row)
    minCol=np.min(col)
    maxCol=np.max(col)

    data=np.zeros((maxRow+1-minRow,maxCol+1-minCol))
    for rowI,colI,value in zip(row,col,islands): data[rowI-minRow][colI-minCol]=value

    return data

def rotate45(data):
    numRows=len(data)
    numCols=len(data[0])
    sideLength=numRows+numCols-1

    rotated=np.zeros((sideLength,sideLength))
    rotated[:,:]=None

    for (rowI, row) in enumerate(data):
        for(colI, val) in enumerate(row):
            outRow=numCols-colI+rowI-1
            outCol=colI+rowI
            try:
                rotated[outRow][outCol]=val
            except Exception:
                pass

    return list(rotated)

def fixOrientation(data):
    for (rowI, row) in enumerate(data):
        for(colI, val) in enumerate(row):
            if(rowI%2==0):
                continue
            else:
                data[rowI][colI]=-val
    return data

def exportToString(data):
    string="first row vertical\n"
    for row in data:
        for cell in row:
            if(np.isnan(cell)):
                string+=" "
            elif(cell==0):
                string+=" "
            else:
                string+=str(int(cell))
            string+=","
        string+="\n"

    return string

def rotatePEEM(row, col, islands):
    islandData=coordsToArray(row,col,islands)
    #islandData=np.array([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17]])
    islandData=rotate45(islandData)
    islandData=fixOrientation(islandData)
    return exportToString(islandData)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Santa fe csv rotate')
    parser.add_argument("file", type=str, help="Path to csv file")
    args=parser.parse_args()

    try:
        file=open(args.file,newline="\n")
    except:
        raise Exception("Error with file")

    with open(args.file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data=[row for row in data if len(row)>1]#remove empty rows
    data=[[int(float(cell)) for cell in row] for row in data]#convert to int


    row=[row[0] for row in data]
    col=[row[1] for row in data]
    islands=[row[2] for row in data]


    with open("rotated.csv","w") as file:
        file.write(rotatePEEM(row,col,islands))
