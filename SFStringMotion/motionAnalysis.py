import json
import rotatePEEM
import santafeAnalysis
import csv
import argparse
import cv2
import numpy as np
#cleanedPEEMFiles/Nov_2020_adjust_/Nov_2020_adjust/800short/800short_330K#_SF_adjustspin_clustering_total.csv 


with open('SFVertexEnergies.json') as f:
  vertexEnergies = json.load(f)


#Returns a string (eg "Sep_2016_600") based on a file path. Used for determining vertex energy dictionary 
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
        lattice=genLattice(rotated,vertexEnergy)

        if lattice.isBlank():
            lattice=None
        
        lattices.append(lattice)
    return lattices


def genLattice(data,vertexEnergy):


    return santafeAnalysis.SantaFeLattice(data, removeEdgeStrings=False, autoAlignCenters=True, vertexEnergies=vertexEnergy)

        



def generateSharedPointGraph(oldLattice,newLattice,oldStrings,newStrings,minSharedPoints=1,minSharedPointsFraction=0,nonInteriorCenterMinSharedPoints=1):
    oldData=[(string,[]) for string in oldStrings]
    newData=[(string, []) for string in newStrings]


    for oldI, oldDataPoint in enumerate(oldData):
        oldString=oldDataPoint[0]
        oldConnections=oldDataPoint[1]

        oldPoints=oldString.getPoints()
        nonInteriorCenterOldPoints=[point for point in oldPoints if oldLattice.data[point[0]][point[1]].interiorCenter==False]#remove interior centers
        oldPoints=set(oldPoints)
        for newI, newDataPoint in enumerate(newData):
            newString=newDataPoint[0]
            newConnections=newDataPoint[1]

            

            newPoints=newString.getPoints()
            nonInteriorCenterNewPoints=[point for point in newPoints if newLattice.data[point[0]][point[1]].interiorCenter==False]#remove interior centers
            newPoints=set(newPoints)
            
            sharedPoints=len(oldPoints.intersection(newPoints))
            sharedNonInteriorCenterPoints=len(set(nonInteriorCenterNewPoints).intersection(set(nonInteriorCenterOldPoints)))


            if sharedPoints>=minSharedPoints and sharedNonInteriorCenterPoints >= nonInteriorCenterMinSharedPoints and (sharedPoints>=len(oldPoints)*minSharedPointsFraction or sharedPoints>=len(newPoints)*minSharedPointsFraction):
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

def groupByTouchingInteriors(oldLattice,newLattice,oldStrings,newStrings):

    groups={}

    for string in oldStrings:
        touching=frozenset(oldLattice.getTouchingCenters(string))
        thisHash=hash(touching)
        if thisHash in groups:
            groups[thisHash][0].append(string)
        else:
            groups[thisHash]=([string],[])
    
    for string in newStrings:
        touching=frozenset(newLattice.getTouchingCenters(string))
        thisHash=hash(touching)
        if thisHash in groups:
            groups[thisHash][1].append(string)
        else:
            groups[thisHash]=([],[string])
    


    return groups.values()


def getNoChange(oldLattice,newLattice,oldStrings,newStrings):
    noChanges=[]
    pointSets={}

    for oldString in [_ for _ in oldStrings]:
        pointSets[hash(frozenset(oldString.getPoints()))]=oldString

    for newString in [_ for _ in newStrings]:
        key=hash(frozenset(newString.getPoints()))
        if key in pointSets.keys():
            oldString=pointSets[key]

            noChanges.append((oldString,newString))
            oldStrings.remove(oldString)
            newStrings.remove(newString)


    return noChanges

       

def getSimpleMotions(oldLattice,newLattice,oldStrings,newStrings):
    groups=groupByTouchingInteriors(oldLattice,newLattice,oldStrings,newStrings)
    motions={
        "grow":[],
        "shrink":[],
        "wiggleEnergyIncrease":[],
        "wiggleEnergyDecrease":[],
        "wiggleEnergySame":[],
        "ambiguousTrivial":[],
        "ambiguousNonTrivial":[],
        "discarded":[]
    }

    for group in groups:


        #check if there are 0 or 1 elements only
        hasLessThanTwoInteriors=False
        if len(group[0])>0 and len(oldLattice.getTouchingCenters(group[0][0]))<2 \
            or len(group[1])>0 and len(newLattice.getTouchingCenters(group[1][0]))<2:
            hasLessThanTwoInteriors=True
        #remove strings that touch less than two interiors which are not loops
        if hasLessThanTwoInteriors:
            for string in group[0]:
                if not string.isLoop():
                    motions["discarded"].append((string,None))
                    oldStrings.remove(string)
            for string in group[1]:
                if not string.isLoop():
                    motions["discarded"].append((None,string))
                    newStrings.remove(string)
            


        
        #1-to-1 string associations
        elif len(group[0])==1 and len(group[1])==1:
            oldString=group[0][0]
            newString=group[1][0]
            oldStringPoints=set(oldString.getPoints())
            newStringPoints=set(newString.getPoints())

            if len(oldStringPoints)<len(newStringPoints):
                motions["grow"].append((oldString,newString))
                
            elif len(oldStringPoints)>len(newStringPoints):
                motions["shrink"].append((oldString,newString))
            else:
                oldEnergy=oldLattice.getStringEnergy(oldString)
                newEnergy=newLattice.getStringEnergy(newString)
                if oldEnergy>newEnergy:
                    motions["wiggleEnergyDecrease"].append((oldString,newString))
                elif oldEnergy<newEnergy:
                    motions["wiggleEnergyIncrease"].append((oldString,newString))
                else:
                    motions["wiggleEnergySame"].append((oldString,newString))
            
            oldStrings.remove(oldString)
            newStrings.remove(newString)

        #n-to-m string associations
        elif len(group[0])>=1 and len(group[1])>=1:
            oldStringCount=len(group[0])
            newStringCount=len(group[1])

            motions["ambiguousTrivial"]+=[group]*min(oldStringCount,newStringCount)
            motions["ambiguousNonTrivial"]+=[group]*(max(oldStringCount,newStringCount)-min(oldStringCount,newStringCount))

            for string in group[0]:
                oldStrings.remove(string)
            for string in group[1]:
                newStrings.remove(string)

    return motions

def isLoop(string,lattice):
    return string.isLoop() and len(lattice.getTouchingCenters(string))<2

def isZ2(string,lattice):
    return len(string.getPoints())==3 and len(lattice.getTouchingCenters(string))==2

def getComplexMotions(oldLattice,newLattice,oldStrings,newStrings):
    oldData,newData=generateSharedPointGraph(oldLattice,newLattice,oldStrings,newStrings)
    graphs=sepererateGraph(oldData,newData)
    
    motions={
        "loopWiggle":[],
        "adjacentReconnection":[],
        "nonTrivialWiggle":[],
        "merge":[],
        "split":[],
        "loopCreation":[],
        "loopAnnihilation":[],
        "z2Creation":[],
        "z2Annihilation":[],
        "creation":[],
        "annihilation":[],
        "complex":[],
        "reconnection":[]
    }

    for graph in graphs:
        old=graph[0]
        new=graph[1]

        graphOldStrings=[i[0] for i in old]
        graphNewStrings=[i[0] for i in new]

        if len(graphOldStrings)==1 and len(graphNewStrings)==1:#grow wiggle shrink
            oldString=graphOldStrings[0]
            newString=graphNewStrings[0]

            if isLoop(oldString, oldLattice) and isLoop(newString,newLattice):
                motions["loopWiggle"].append((oldString,newString))
            elif set(oldLattice.getTouchingCompositeSquares(oldString))==set(newLattice.getTouchingCompositeSquares(newString)) and len(set(newLattice.getTouchingCompositeSquares(newString)))>0:

                motions["adjacentReconnection"].append((oldString,newString))
            else:
                motions["nonTrivialWiggle"].append((oldString,newString))
        elif len(old)==1 and len(new)>1:
            motions["split"].append((graphOldStrings[0],graphNewStrings))
        elif len(old)>1 and len(new)==1:
            motions["merge"].append((graphOldStrings,graphNewStrings[0]))
        elif len(new)==2 and len(old)==2 and len(new[0][1])==2 and len(new[1][1])==2 and len(old[0][1])==2 and len(old[1][1])==2:
            motions["reconnection"].append((graphOldStrings,graphNewStrings))
        elif len(old)==0 and len(new)==1:
            newString=graphNewStrings[0]
            if isLoop(newString,newLattice):
                motions["loopCreation"].append((None,newString))
            elif isZ2(newString,newLattice):
                motions["z2Creation"].append((None,newString))
            else:
                motions["creation"].append((None,newString))
        elif len(old)==1 and len(new)==0:
            oldString=graphOldStrings[0]
            if isLoop(oldString,oldLattice):
                motions["loopAnnihilation"].append((oldString,None))
            elif isZ2(oldString,oldLattice):
                motions["z2Annihilation"].append((oldString,None))
            else:
                motions["annihilation"].append((oldString,None))
        else:
            motions["complex"].append((graphOldStrings,graphNewStrings))

    return motions

def getBaseMotionData(oldLattice,newLattice,oldStrings,newStrings):
    
    oldData,newData=generateSharedPointGraph(oldLattice,newLattice,oldStrings,newStrings)


    graphs=sepererateGraph(oldData,newData)
    
    noChange=[]

    wiggle=[]
    grow=[]
    shrink=[]


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
            oldString=old[0][0]
            newString=new[0][0]
            oldPoints=oldString.getPoints()
            newPoints=newString.getPoints()
            oldPointCount=len(oldPoints)
            newPointCount=len(newPoints)
            
            motionElement=([oldString],[newString])

            if oldPointCount==newPointCount:#wiggle or noChange
                if set(oldPoints)==set(newPoints):
                    noChange.append(motionElement)
                else:
                    wiggle.append(motionElement)
            if oldPointCount>newPointCount:
                shrink.append(motionElement)
            elif newPointCount>oldPointCount:
                grow.append(motionElement)

        elif len(old)==1 and len(new)==0:
            deletion.append(    ([old[0][0]],[])    )
        elif len(old)==0 and len(new)==1:
            creation.append(    ([],[new[0][0]])    )
        elif len(old)==1 and len(new)>=2:
            split.append(   ([old[0][0]],[el[0] for el in new])   )
        elif len(new)==1 and len(old)>=2:
            merge.append(   ([el[0] for el in old], [new[0][0]])  )
        elif len(new)==2 and len(old)==2 and len(new[0][1])==2 and len(new[1][1])==2 and len(old[0][1])==2 and len(old[1][1])==2:
            #there are two elements in new and old and each element has two connections
            reconnection.append([[el[0] for el in old],[el[0] for el in new]])
        else:
            complex.append([[el[0] for el in old],[el[0] for el in new]])

        
    return {
        "noChange":noChange,
        "wiggle":wiggle,
        "grow":grow,
        "shrink":shrink,
        "creation":creation,
        "deletion":deletion,
        "merge":merge,
        "split":split,
        "reconnection":reconnection,
        "complex":complex
    }

def splitTrivialNonTrivail(motions,oldLattice,newLattice,keys=None):
    if keys==None:
        keys=[key for key in motions.keys()]
    
    for key in keys:
        motions["trivial "+key]=[]
        motions["nonTrivial "+key]=[]
        for motion in motions[key]:
            oldStrings=motion[0]
            newStrings=motion[1]
            oldInteriors=[]
            newInteriors=[]
            for oldString in oldStrings: oldInteriors.append(oldLattice.getTouchingInteriors(oldString))
            for newString in newStrings: newInteriors.append(newLattice.getTouchingInteriors(newString))
            if sorted(oldInteriors)==sorted(newInteriors):
                motions["trivial "+key].append(motion)
            else:
                motions["nonTrivial "+key].append(motion)
        del motions[key]
    
    return motions


def stringifyChanges(changes):
    if changes is None:
        return ""
    

    string=""
    for type,occurences in changes.items():
        if type=="noChange" :
            string+=f"{len(occurences)} strings: no Change\n"
            continue
        if type=="discarded":
            string+=f"{len(occurences)} strings: discarded\n"
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

    noChange={"noChange":getNoChange(oldLattice,newLattice,oldStrings,newStrings)}
    simpleMotions=getSimpleMotions(oldLattice,newLattice,oldStrings,newStrings)
    complexMotions=getComplexMotions(oldLattice,newLattice,oldStrings,newStrings)

    motions={**noChange,**simpleMotions,**complexMotions}

    """motions=getBaseMotionData(oldLattice,newLattice,oldStrings,newStrings)
    motions=splitTrivialNonTrivail(motions,oldLattice,newLattice,keys=["wiggle","grow","shrink"])

    

    #add energy levels for wiggle
    motions["nonTrivialWiggleEnergyDecrease"], motions["nonTrivialWiggleEnergySame"], motions["nonTrivialWiggleEnergyIncrease"] = classifyWiggleEnergyChange(motions["trivial wiggle"],oldLattice,newLattice)
    motions["trivialWiggleEnergyDecrease"], motions["trivialWiggleEnergySame"], motions["trivialWiggleEnergyIncrease"] = classifyWiggleEnergyChange(motions["nonTrivial wiggle"],oldLattice,newLattice)
    del motions["trivial wiggle"]
    del motions["nonTrivial wiggle"]

    #add loop 
    for motionType in ["creation","deletion","trivialWiggleEnergyIncrease","trivialWiggleEnergySame","trivialWiggleEnergyDecrease","nonTrivialWiggleEnergyDecrease","nonTrivialWiggleEnergySame","nonTrivialWiggleEnergyIncrease","trivial grow","trivial shrink","nonTrivial grow","nonTrivial shrink"]:
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
    """

    

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



#gets raw santa fe data from a file
def getSFMotionFileData(fileName):
    try:
        file=open(fileName,newline="\n")
    except:
        raise Exception("Error with file")

    with open(fileName, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


#returns an array of with lattices in it based on raw file data
def getLatticesFromRawData(data,vertexEnergy,maxIndex=None):
    data=[row for row in data if len(row)>1]#remove empty rows
    data=[[int(float(cell)) for cell in row] for row in data]#convert to int

    #turn columns into rows
    rowData=[row[0] for row in data]
    colData=[row[1] for row in data]
    islandData=[[row[i] for row in data] for i in range(2,len(data[0]))]

    lattices=[]

    if maxIndex:
        islandData=islandData[0:maxIndex]

    for index,islands in enumerate(islandData):
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=genLattice(rotated,vertexEnergy)

        if lattice.isBlank():
            lattice=None
        
        lattices.append(lattice)
    return lattices

class SFMotionSample:
    def __init__(self,data,fileType,skippedIndex=None,maxIndex=None,fileName="unknown file"):
        self.rawData=data
        self.skippedIndex=skippedIndex
        self.maxIndex=maxIndex
        self.fileName=fileName
        self.fileType=fileType
        self.vertexEnergies=vertexEnergies[fileType]

        self.lattices=getLatticesFromRawData(data,self.vertexEnergies,maxIndex=maxIndex)
        self.fixLatticeHoles()

        self.savedMotions=self.getMotions()
        self.savedImages=self.getImages(motions=self.savedMotions)

    #fills in missing lattice moments with the most recently known moment state
    def fixLatticeHoles(self):
        lattices=self.lattices

        #set the base lattice data 
        i=0
        while lattices[i] is None: i+=1
        mostRecentLatticeData=lattices[i].rawData

        #loop through all non-None lattices
        for i,lattice in enumerate(lattices):
            if lattice is None:
                continue

            correctedCellCoords=[]

            #check each cell of the data
            latticeData=lattice.rawData
            for (rowI,row) in enumerate(latticeData):
                for (colI,cell) in enumerate(row):
                    #if the cell is non None update the most recent lattice data
                    if cell is not None:
                        mostRecentLatticeData[rowI][colI]=cell
                    
                    #if the cell is none and can be corrected
                    if cell is None and mostRecentLatticeData[rowI][colI] is not None:
                        latticeData[rowI][colI]=mostRecentLatticeData[rowI][colI]
                        correctedCellCoords.append((rowI,colI))
                        #print(f"fixed hole on slide {i}: {rowI},{colI}")
            
            lattice.rawData=latticeData
            lattice.updateFromRawDataChange()

            #mark as corrected
            for coord in correctedCellCoords:
                lattice.data[coord[0]][coord[1]].oldData=True
    
    def getMotions(self):
        lattices=self.lattices
        motions=[None]
        for index, lattice in enumerate(self.lattices[:-1]):
            #print(f"frame {index}->{index+1}")

            thisLattice=lattice
            nextLattice=lattices[index+1]
            
            if self.skippedIndex is not None and (index+1)%self.skippedIndex==0 :
                motions.append(None)
            elif thisLattice==None or nextLattice==None or thisLattice.isBlank() or nextLattice.isBlank():
                motions.append(None)
            else:
                motions.append(getMotions(thisLattice,nextLattice))

        return motions
    def getImages(self,motions=None):
        images=[]

        for i, lattice in enumerate(self.lattices):
            
            outputImage=np.zeros((1000,1000,3), np.uint8)
            outputImage[:,:]=(250,250,250)
            if lattice!=None:
                lattice.drawStrings(outputImage,lineWidth=3,showID=True)
                lattice.drawCells(outputImage)

            if motions is not None:
                outputImage=writeMotionText(motions[i],outputImage)
            else:
                blank=np.zeros((1000,500,3), np.uint8)
                blank[:,:]=(250,250,250)
                #outputImage=np.concatenate((outputImage,blank), axis=1)
            
            cv2.putText(outputImage,self.fileName,(0,20), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(outputImage,"frame "+str(i),(0,40), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
            
            images.append(outputImage)
        return images
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Peem analysis')
    parser.add_argument("file", type=str, help="Path to csv file")

    args=parser.parse_args()

    fileType=determineFileType(args.file)
    data=getSFMotionFileData(args.file)
    motion=SFMotionSample(data,fileType,fileName=args.file,skippedIndex=10,maxIndex=None)

    images=motion.savedImages
    for i,image in enumerate(images):
        cv2.imwrite(f"out/{i}.png",np.float32(image))

    i=0;
    while True:

        cv2.imshow("window",images[i])
        key=cv2.waitKey(0)
        if key==13:
            break
        elif key==81 or key==ord("a"):
            i-=1
            if i<0: i=len(images)-1
        elif key==83 or key==ord("d"):
            i=(i+1)%len(images)
        else:
            print(key)
            
