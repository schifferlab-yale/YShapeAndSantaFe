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


def classifyWiggleEnergyChange(wiggles,oldLattice,newLattice):
    decrease,same,increase=[],[],[]
    for wiggle in wiggles:
        oldEnergy=oldLattice.getStringEnergy(wiggle[0][0])
        newEnergy=newLattice.getStringEnergy(wiggle[1][0])
        if oldEnergy>newEnergy:
            decrease.append(wiggle)
        elif newEnergy>oldEnergy:
            increase.append(wiggle)
        else:
            same.append(wiggle)

    return decrease,same,increase


def rotatedFileDataAsSantaFe(fileName,maxIndex=None):
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

    vertexEnergy=vertexEnergies[determineFileType(fileName)]

    if maxIndex:
        islandData=islandData[0:maxIndex]

    for index,islands in enumerate(islandData):
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=santafeAnalysis.SantaFeLattice(rotated, removeEdgeStrings=False, autoAlignCenters=True, vertexEnergies=vertexEnergy)

        if lattice.isBlank():
            lattice=None
        
        lattices.append(lattice)
    return lattices

def groupByTouchingInteriors(oldLattice,newLattice,oldStrings,newStrings):

    groups={}

    for string in oldStrings:
        touching=frozenset(oldLattice.getTouchingCompositeSquares(string))
        if(len(touching)==0):
            continue
        thisHash=hash(touching)
        if thisHash in groups:
            groups[thisHash][0].append(string)
        else:
            groups[thisHash]=([string],[])
    
    for string in newStrings:
        touching=frozenset(newLattice.getTouchingCompositeSquares(string))
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


#this will edit the arrays
def getTrivialMotions(oldLattice,newLattice,oldStrings,newStrings,minExtraSharedPoints=1):


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
            trivialAmbiguous+=[[[None],[None]]]*newCount
            deletion+=[[[None],[]]]*(oldCount-newCount)
        elif newCount>oldCount:#a creation occured
            trivialAmbiguous+=[[[None],[None]]]*oldCount
            creation+=[[[],[None]]]*(newCount-oldCount)
        elif newCount==oldCount:
            if newCount>1:
                trivialAmbiguous+=[[[None],[None]]]*newCount
            elif newCount==1:
                oldString=group[0][0]
                newString=group[1][0]
                oldLength=len(oldString.getPoints())
                newLength=len(newString.getPoints())

                oldStringPoints = set(oldString.getNonInteriorCenterPoints(oldLattice))
                newStringPoints = set(newString.getNonInteriorCenterPoints(newLattice))
                numExtraSharedPoints=len(oldStringPoints.intersection(newStringPoints))

                if numExtraSharedPoints>=minExtraSharedPoints:
                    if(oldLength>newLength):
                        trivialShrink.append(group)
                    elif newLength>oldLength:
                        trivialGrow.append(group)
                    else:
                        trivialWiggle.append(group)
                else:
                    continue
                    #continue so we don't remove these strings
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


def generateSharedPointGraph(oldLattice,newLattice,oldStrings,newStrings):
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
            newPoints=[point for point in newPoints if newLattice.data[point[0]][point[1]].interiorCenter==False]#remove interior centers
            newPoints=set(newPoints)

            if len(oldPoints.intersection(newPoints))>0:
                oldConnections.append(newDataPoint)
                newConnections.append(oldDataPoint)

    return oldData, newData

def sepererateGraph(top,bottom):
    graphs=[]
    topCategorized=[]
    bottomCategorized=[]

    for el in top:
        if el not in topCategorized:
            topGroup,bottomGroup=traverseGraph(top,bottom,True,el,[],[])
            topCategorized+=topGroup
            bottomCategorized+=bottomGroup
            graphs.append((topGroup,bottomGroup))
    
    for el in bottom:
        if el not in bottomCategorized:
            topGroup,bottomGroup=traverseGraph(top,bottom,False,el,[],[])
            topCategorized+=topGroup
            bottomCategorized+=bottomGroup
            graphs.append((topGroup,bottomGroup))

    return graphs

def traverseGraph(top,bottom,onTop,currEl,topSeen,bottomSeen):
    topSeen=[el for el in topSeen]
    bottomSeen=[el for el in bottomSeen]

    currArray=top if onTop else bottom
    currSeenArray=topSeen if onTop else bottomSeen

    if currEl in currSeenArray:
        return topSeen,bottomSeen
    else:
        currSeenArray.append(currEl)

    #base case
    if len(currEl[1])==0:
        return topSeen, bottomSeen
    
    else:
        for el in currEl[1]:
            topSeen,bottomSeen = traverseGraph(top,bottom,not onTop, el, topSeen,bottomSeen)
        return topSeen, bottomSeen


    
def getNonTrivialMotions(oldLattice,newLattice,oldStrings,newStrings):
    
    oldData,newData=generateSharedPointGraph(oldLattice,newLattice,oldStrings,newStrings)


    graphs=sepererateGraph(oldData,newData)
    
    nonTrivialWiggle=[]
    nonTrivialGrow=[]
    nonTrivialShrink=[]

    trivialGrow=[]
    trivialWiggle=[]
    trivialShrink=[]

    creation=[]
    deletion=[]

    merge=[]
    split=[]

    reconnection=[]

    complex=[]

    for graph in graphs:
        old=graph[0]
        new=graph[1]


        if len(old)==1 and len(new)==1:#grow wiggle shrink
            oldPointCount=len(old[0][0].getPoints())
            newPointCount=len(new[0][0].getPoints())

            touchingNoInteriors=len(oldLattice.getTouchingInteriors(old[0][0]))==0 and len(newLattice.getTouchingInteriors(new[0][0]))==0

            if oldPointCount>newPointCount:
                if touchingNoInteriors:
                    trivialShrink.append([[old[0][0]],[new[0][0]]])
                else:
                    nonTrivialShrink.append([[old[0][0]],[new[0][0]]])
            elif newPointCount>oldPointCount:
                if touchingNoInteriors:
                    trivialGrow.append([[old[0][0]],[new[0][0]]])
                else:
                    nonTrivialGrow.append([[old[0][0]],[new[0][0]]])
            else:
                if touchingNoInteriors:
                    trivialWiggle.append([[old[0][0]],[new[0][0]]])
                else:
                    nonTrivialWiggle.append([[old[0][0]],[new[0][0]]])
        elif len(old)==1 and len(new)==0:
            deletion.append([[old[0][0]],[]])
        elif len(old)==0 and len(new)==1:
            creation.append([[],[new[0][0]]])
        elif len(old)==1 and len(new)>=2:
            split.append([old[0][0],[el[0] for el in new]])
        elif len(new)==1 and len(old)>=2:
            merge.append([[el[0] for el in old], new[0][0]])
        elif len(new)==2 and len(old)==2 and len(new[0][1])==2 and len(new[1][1])==2 and len(old[0][1])==2 and len(old[1][1])==2:
            #there are two elements in new and old and each element has two connections
            reconnection.append([[el[0] for el in old],[el[0] for el in new]])
        else:
            complex.append([[el[0] for el in old],[el[0] for el in new]])

        




    return {
        "nonTrivialWiggle":nonTrivialWiggle,
        "nonTrivialGrow":nonTrivialGrow,
        "nonTrivialShrink":nonTrivialShrink,
        "trivialGrow":trivialGrow,
        "trivialWiggle":trivialWiggle,
        "trivialShrink":trivialShrink,
        "creation":creation,
        "deletion":deletion,
        "merge":merge,
        "split":split,
        "reconnection":reconnection,
        "complex":complex
    }



def stringifyChanges(changes):

    string=""
    for type,occurences in changes.items():
        if type=="noChange":
            string+=f"{len(occurences)} strings: no Change\n"
            continue
        for occurence in occurences:
            string+=f"{occurence[0]}->{occurence[1]} {type}\n"

    """
    string=""
    string+=f"{len(changes['noChange'])}: no change\n"
    string+=f"{len(changes['ambiguousCreation'])}: ambiguous creation\n"
    string+=f"{len(changes['ambiguousDeletion'])}: ambiguous deletion\n"
    string+=f"{len(changes['trivialAmbiguous'])}: ambiguous trivial\n"
    #for motion in changes["trivialAmbiguous"]:
        #string+=f"{motion[0]}->{motion[1]} trivial ambiguous\n"
    for motion in changes["trivialGrow"]:
        string+=f"{motion[0]}->{motion[1]} trivial grow\n"
    for motion in changes["trivialWiggleEnergyDecrease"]:
        string+=f"{motion[0]}->{motion[1]} trivial wiggle energy decrease\n"
    for motion in changes["trivialWiggleEnergySame"]:
        string+=f"{motion[0]}->{motion[1]} trivial wiggle energy same\n"
    for motion in changes["trivialWiggleEnergyIncrease"]:
        string+=f"{motion[0]}->{motion[1]} trivial wiggle energy increase\n"
    for motion in changes["trivialShrink"]:
        string+=f"{motion[0]}->{motion[1]} trivial shrink\n"
    
    string+="\n\n"

    for motion in changes["nonTrivialGrow"]:
        string+=f"{motion[0]}->{motion[1]} non trivial grow\n"
    for motion in changes["nonTrivialWiggleEnergyDecrease"]:
        string+=f"{motion[0]}->{motion[1]} non trivial wiggle energy decrease\n"
    for motion in changes["nonTrivialWiggleEnergySame"]:
        string+=f"{motion[0]}->{motion[1]} non trivial wiggle energy same\n"
    for motion in changes["nonTrivialWiggleEnergyIncrease"]:
        string+=f"{motion[0]}->{motion[1]} non trivial wiggle energy increase\n"
    for motion in changes["nonTrivialShrink"]:
        string+=f"{motion[0]}->{motion[1]} non trivial shrink\n"

    for motion in changes["creation"]:
        string+=f"{motion} creation\n"
    for motion in changes["deletion"]:
        string+=f"{motion} deletion\n"

    for motion in changes["merge"]:
        string+=f"{motion[0]}->{motion[1]} merge\n"
    for motion in changes["split"]:
        string+=f"{motion[0]}->{motion[1]} split\n"
    for motion in changes["reconnection"]:
        string+=f"{motion[0]}->{motion[1]} reconnection\n"
    for motion in changes["complex"]:
        string+=f"{motion[0]}->{motion[1]} complex motion\n"
        
    """

    return string

def getMotions(oldLattice,newLattice):
    oldStrings=[string for string in oldLattice.strings]
    newStrings=[string for string in newLattice.strings]

    trivialMotions=getTrivialMotions(oldLattice,newLattice,oldStrings,newStrings)

    nonTrivialMotions=getNonTrivialMotions(oldLattice,newLattice,oldStrings,newStrings)

    

    #add energy levels for wiggle
    motions = {}
    for motionName in set(trivialMotions.keys()) | set(nonTrivialMotions.keys()):
        motions[motionName]=[]
        if motionName in trivialMotions.keys():
            motions[motionName]+=trivialMotions[motionName]
        if motionName in nonTrivialMotions.keys():
            motions[motionName]+=nonTrivialMotions[motionName]
    motions["nonTrivialWiggleEnergyDecrease"], motions["nonTrivialWiggleEnergySame"], motions["nonTrivialWiggleEnergyIncrease"] = classifyWiggleEnergyChange(motions["nonTrivialWiggle"],oldLattice,newLattice)
    motions["trivialWiggleEnergyDecrease"], motions["trivialWiggleEnergySame"], motions["trivialWiggleEnergyIncrease"] = classifyWiggleEnergyChange(motions["trivialWiggle"],oldLattice,newLattice)
    del motions["trivialWiggle"]
    del motions["nonTrivialWiggle"]

    #add loop 
    for motionType in ["creation","deletion","trivialWiggleEnergyIncrease","trivialWiggleEnergySame","trivialWiggleEnergyDecrease","nonTrivialWiggleEnergyDecrease","nonTrivialWiggleEnergySame","nonTrivialWiggleEnergyIncrease","trivialGrow","trivialShrink","nonTrivialGrow","nonTrivialShrink"]:
        motions[motionType+"-loop"]=[]
        for motionPair in motions[motionType]:
            valid=True
            for string in motionPair[0]+motionPair[1]:
                if not string.isLoop() or len(string.getInteriorCenterPoints(oldLattice))>=2:
                    valid=False
                    break
            if valid:
                motions[motionType+"-loop"].append(motionPair)
        motions[motionType]=[el for el in motions[motionType] if el not in motions[motionType+"-loop"]]#delete the ones from the main list
        


    

    return motions

def analyzeMotion(lattices,skippedIndex=None,numFrames=-1):
    motions=[None]
    for index, lattice in enumerate(lattices[:-1]):

        thisLattice=lattice
        nextLattice=lattices[index+1]
        
        if skippedIndex is not None and (index+1)%skippedIndex==0 :
            motions.append(None)
        elif thisLattice==None or nextLattice==None or thisLattice.isBlank() or nextLattice.isBlank():
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
            cv2.putText(frame, outLine, (1000,40+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)

    

    return frame


def analyzeFile(fileName,skippedIndex=None,maxIndex=None):
    lattices=rotatedFileDataAsSantaFe(fileName,maxIndex=maxIndex)
    if maxIndex:
        lattices=lattices[0:maxIndex]

    motionTotals={"analyzedFrames":0}
    images=[]

    motions=analyzeMotion(lattices,skippedIndex=skippedIndex)
    for i, (lattice, frameMotions) in enumerate(zip(lattices,motions)):
        if frameMotions is not None:
            motionTotals["analyzedFrames"]+=1
            for key in frameMotions.keys():
                if key not in motionTotals:
                    motionTotals[key]=len(frameMotions[key])
                else:
                    motionTotals[key]+=len(frameMotions[key])
        
        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        if lattice!=None:
            lattice.drawStrings(outputImage,lineWidth=3,showID=True)
            lattice.drawCells(outputImage)

            outputImage=writeMotionText(frameMotions,outputImage)
        else:
            #make sure its still the same dimensions
            blank=np.zeros((1000,500,3), np.uint8)
            blank[:,:]=(250,250,250)
            outputImage=np.concatenate((outputImage,blank), axis=1)
        images.append(outputImage)
        cv2.putText(outputImage,fileName,(0,20), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(outputImage,"frame "+str(i),(0,40), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
        
    
    return motionTotals, images, lattices



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Peem analysis')
    parser.add_argument("file", type=str, help="Path to csv file")

    args=parser.parse_args()

    motions,images,lattices=analyzeFile(args.file,skippedIndex=10,maxIndex=100)
    print(motions)
    

    i=0;
    while True:


        """outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        if lattices[i]!=None:
            lattices[i].drawStrings(outputImage,lineWidth=3,showID=True)
            lattices[i].drawCells(outputImage)

            outputImage=writeMotionText(motions[i],outputImage)
        cv2.putText(outputImage,"frame "+str(i),(0,30), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
        
        
        cv2.imshow("window",outputImage)
        #cv2.imwrite(f"out/{i}.png",np.float32(outputImage))
        key=cv2.waitKey(0)
        if key==13:
            break
        elif key==81:
            i-=1
        else:
            i+=1"""

        cv2.imshow("window",images[i])
        key=cv2.waitKey(0)
        if key==13:
            break
        elif key==81 or key==ord("a"):
            i-=1
        elif key==83 or key==ord("d"):
            i+=1
        else:
            print(key)
            
