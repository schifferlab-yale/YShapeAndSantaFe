import json
import rotatePEEM
import santafeAnalysis
import csv
import argparse
import cv2
import numpy as np



with open('SFVertexEnergies.json') as f:
  vertexEnergies = json.load(f)

def determineFileType(path):
    path=path.lower()
    if "sep_2016" in path:
        if "600" in path:
            return "Sep_2016_600"
        elif "700" in path:
            return "Sep_2016_700"
        elif "800" in path:
            return "Sep_2016_800"
    if "mar_2018" in path:
        if "600" in path:
            return "Mar_2018_600"
    elif "may_2017" in path:
        if "700" in path:
            return "May_2017_700"
        elif "800" in path:
            return "May_2017_800"
    elif "nov_2019" in path:
        if "600" in path:
            return "Nov_2019_600"
        elif "700" in path:
            return "Nov_2019_700"
        elif "800" in path:
            return "Nov_2019_800"
    elif "nov_2020" in path:
        if "600" in path:
            return "Nov_2020_600"
        elif "700" in path:
            return "Nov_2020_700"
        elif "800" in path:
            return "Nov_2020_800"
    
    raise Exception("Could not determine "+path)


def rotatedFileDataAsSantaFe(fileName):
    try:
        file=open(fileName,newline="\n")
    except:
        raise Exception("Error with file")

    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data=[row for row in data if len(row)>1]#remove empty rows
    data=[[int(float(cell)) for cell in row] for row in data]#convert to int

    #turn columns into rows
    rowData=[row[0] for row in data]
    colData=[row[1] for row in data]
    islandData=[[row[i] for row in data] for i in range(2,len(data[0]))]

    lattices=[]

    vertexEnergies=determineFileType(fileName)

    for index,islands in enumerate(islandData):
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=santafeAnalysis.SantaFeLattice(rotated, removeEdgeStrings=False, autoAlignCenters=True, vertexEnergies=vertexEnergies)

        if lattice.isBlank():
            lattice=None
        
        lattices.append(lattice)
    return lattices

def groupByTouchingInteriors(oldLattice,newLattice,oldStrings,newStrings):

    groups={}

    for string in oldStrings:
        touching=frozenset(oldLattice.getTouchingCenters(string))
        if(len(touching)==0):
            continue
        thisHash=hash(touching)
        if thisHash in groups:
            groups[thisHash][0].append(string)
        else:
            groups[thisHash]=([string],[])
    
    for string in newStrings:
        touching=frozenset(oldLattice.getTouchingCenters(string))
        if(len(touching)==0):
            continue
        thisHash=hash(touching)
        if thisHash in groups:
            groups[thisHash][1].append(string)
        else:
            groups[thisHash]=([],[string])
    


    return groups.values()

    

        
def getNoChange(oldStrings,newStrings):
    noChanges=[]

    for string in oldStrings:
        points=set(string.getPoints())
        for string2 in newStrings:
            if set(string2.getPoints())==points:
                noChanges.append((string,string2))
                break

    return noChanges

def getTrivialMotions(oldLattice,newLattice,oldStrings,newStrings):


    #get no changes and remove
    noChange=getNoChange(oldStrings,newStrings)
    for pair in noChange:
        oldStrings.remove(pair[0])
        newStrings.remove(pair[1])
    
    #main part
    groups=groupByTouchingInteriors(oldLattice,newLattice,oldStrings,newStrings)
    #remove non-paired strings
    groups=[group for group in groups if len(group[0])!=0 and len(group[1])!=0]
    
    #classify remaining trivial motions and remove from list
    trivialAmbiguous=[]
    trivialGrow=[]
    trivialWiggle=[]
    trivialShrink=[]
    deletion=[]
    creation=[]
    for group in groups:
        oldCount=len(group[0])
        newCount=len(group[1])
        if oldCount>newCount:#a deletion occured
            trivialAmbiguous+=[None]*newCount
            deletion+=[None]*(oldCount-newCount)
        elif newCount>oldCount:#a creation occured
            trivialAmbiguous+=[None]*oldCount
            creation+=[None]*(newCount-oldCount)
        elif newCount==oldCount:
            if newCount>1:
                trivialAmbiguous+=[None]*newCount
            elif newCount==1:
                oldString=group[0][0]
                newString=group[1][0]
                oldLength=len(oldString.getPoints())
                newLength=len(newString.getPoints())
                if(oldLength>newLength):
                    trivialShrink.append(group)
                elif newLength>oldLength:
                    trivialGrow.append(group)
                else:
                    trivialWiggle.append(group)
            else:
                raise Exception("Should not get here")

        for string in group[0]:
            oldStrings.remove(string)
        for string in group[1]:
            newStrings.remove(string)





    return {
        "noChange":noChange, 
        "trivialAmbiguous":trivialAmbiguous,
        "trivialGrow":trivialGrow,
        "trivialWiggle":trivialWiggle,
        "trivialShrink":trivialShrink,
        "ambiguousCreation":creation,
        "ambiguousDeletion":deletion,
        }


def getNonTrivialMotions(oldLattice,newLattice,oldStrings,newStrings):
    oldData=[(string,[]) for string in oldStrings]
    newData=[(string, []) for string in newStrings]
    for oldI, oldDataPoint in enumerate(oldData):
        oldString=oldDataPoint[0]
        oldConnections=oldDataPoint[1]

        oldPoints=oldString.getPoints()
        oldPoints=[point for point in oldPoints if oldLattice.data[point[0]][point[1]].interiorCenter==False]#remove interior centers
        oldPoints=set(oldPoints)
        for newI, newDataPoint in enumerate(newData):
            newString=newDataPoint[0]
            newConnections=newDataPoint[1]

            newPoints=newString.getPoints()
            newPoints=[point for point in newPoints if oldLattice.data[point[0]][point[1]].interiorCenter==False]#remove interior centers
            newPoints=set(newPoints)

            if len(oldPoints.intersection(newPoints))>0:
                oldConnections.append(newI)
                newConnections.append(oldI)
    
    



def stringifyChanges(changes):
    string=""
    string+=f"{len(changes['noChange'])}: no change\n"
    string+=f"{len(changes['ambiguousCreation'])}: ambiguous creation\n"
    string+=f"{len(changes['ambiguousDeletion'])}: ambiguous deletion\n"
    string+=f"{len(changes['trivialAmbiguous'])}: ambiguous trivial\n"
    #for motion in changes["trivialAmbiguous"]:
        #string+=f"{motion[0]}->{motion[1]} trivial ambiguous\n"
    for motion in changes["trivialGrow"]:
        string+=f"{motion[0]}->{motion[1]} trivial grow\n"
    for motion in changes["trivialWiggle"]:
        string+=f"{motion[0]}->{motion[1]} trivial wiggle\n"
    for motion in changes["trivialShrink"]:
        string+=f"{motion[0]}->{motion[1]} trivial shrink\n"
    

    return string

def getMotions(oldLattice,newLattice):
    oldStrings=[string for string in oldLattice.strings]
    newStrings=[string for string in newLattice.strings]

    motions=getTrivialMotions(oldLattice,newLattice,oldStrings,newStrings)
    return motions

def analyzeMotion(lattices,skippedIndex=None,numFrames=-1):
    motions=[None]
    for index, lattice in enumerate(lattices[:-1]):

        thisLattice=lattice
        nextLattice=lattices[index+1]
        
        if skippedIndex is not None and (index+1)%skippedIndex==0:
            motions.append(None)
        elif thisLattice==None or nextLattice==None:
            motions.append(None)
        else:
            motions.append(getMotions(thisLattice,nextLattice))


        #exit early if needed
        if numFrames>0 and index>numFrames:
            break
    return motions

def writeMotionText(motions,frame):
    blank=np.zeros((1000,500,3), np.uint8)
    blank[:,:]=(250,250,250)
    frame=np.concatenate((frame,blank), axis=1)
    #this splits it into lines
    if(motions is not None):
        lineLength=60
        lines=(stringifyChanges(motions).split("\n"))
        i=0
        while(i<len(lines)):
            line=lines[i]
            while(len(line)>lineLength):
                lines.insert(i,line[:lineLength])
                line=line[lineLength:]
                i+=1
                lines[i]=line


            i+=1
        for i, outLine in enumerate(lines):
            cv2.putText(frame, outLine, (1000,15+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)

    

    return frame

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Peem analysis')
    parser.add_argument("file", type=str, help="Path to csv file")
    args=parser.parse_args()

    lattices=rotatedFileDataAsSantaFe(args.file)

    #DELETE THIS LATER
    #lattices=lattices[0:5]
    
    motions=analyzeMotion(lattices)

    i=0;
    while True:


        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        if lattices[i]!=None:
            lattices[i].drawStrings(outputImage,lineWidth=3,showID=True)
            lattices[i].drawCells(outputImage)

            outputImage=writeMotionText(motions[i],outputImage)
        cv2.putText(outputImage,"frame "+str(i),(0,30), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
        
        
        cv2.imshow("window",outputImage)
        key=cv2.waitKey(0)
        if key==13:
            break
        elif key==81:
            i-=1
        else:
            i+=1
            
