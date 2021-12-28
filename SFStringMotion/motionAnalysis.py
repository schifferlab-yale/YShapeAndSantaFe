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
        "loopGrow":[],
        "loopShrink":[],
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
                oldStringLength=len(oldString.getPoints())
                newStringLength=len(newString.getPoints())
                if oldStringLength>newStringLength:
                    motions["loopShrink"].append((oldString,newString))
                elif oldStringLength<newStringLength:
                    motions["loopGrow"].append((oldString,newString))
                else:
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


    return string

def getNoChangeMotions(oldLattice,newLattice,oldStrings,newStrings):
    motions={"noChangeCenter":[],"noChangeEdge":[]}
    noChangeMotions=getNoChange(oldLattice,newLattice,oldStrings,newStrings)

    for i, (oldString, newString) in enumerate(noChangeMotions):
        if (len(oldLattice.getTouchingCenters(oldString))<2 \
            or len(newLattice.getTouchingCenters(newString))<2) \
            and not oldString.isLoop() and not newString.isLoop():
            motions["noChangeEdge"].append(noChangeMotions[i])
        else:
            motions["noChangeCenter"].append(noChangeMotions[i])
    
    return motions


def getMotions(oldLattice,newLattice):
    oldStrings=[string for string in oldLattice.strings]
    newStrings=[string for string in newLattice.strings]



    noChange=getNoChangeMotions(oldLattice,newLattice,oldStrings,newStrings)



    simpleMotions=getSimpleMotions(oldLattice,newLattice,oldStrings,newStrings)
    complexMotions=getComplexMotions(oldLattice,newLattice,oldStrings,newStrings)

    motions={**noChange,**simpleMotions,**complexMotions}


    

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
        
    
    def getNumStrings(self):
        n=0
        for lattice in self.lattices:
            if lattice is not None:
                n+=len(lattice.strings)
        return n
    def getNumDimers(self):
        n=0
        for lattice in self.lattices:
            if lattice is not None:
                n+=lattice.numDimers()
        return n
    def getMotionSummary(self):
        motionCounts={}
        i=0
        while self.savedMotions[i]==None:
            i+=1
        for key in self.savedMotions[i].keys():
            motionCounts[key]=[]

        
        motionCounts["complexMotionVertexFraction"]=[]

 
        for i, motionSet in enumerate([i for i in self.savedMotions if i is not None]):

            for key in motionCounts.keys():
                motionCounts[key].append(0)

            for type, motions in motionSet.items():
                if type=="complex":
                    enteredStrings=0
                    for motion in motions:
                        enteredStrings+=len(motion[0])
                        for string in motion[0]:
                            motionCounts["complexMotionVertexFraction"][i]+=len(string.getVertexPoints(string.lattice))/string.lattice.numDimers()
                    motionCounts[type][i]+=enteredStrings
                else:
                    motionCounts[type][i]+=len(motions)


        for key in [key for key in motionCounts.keys()]:
            data=motionCounts[key]
            motionCounts[key+"Avg"]=np.average(data)
            motionCounts[key+"Err"]=np.std(data)/np.sqrt(np.size(data))
            del motionCounts[key]
        
        motionCounts["numFrames"]=len([i for i in self.savedMotions if i is not None])
        motionCounts["numStrings"]=self.getNumStrings()
        motionCounts["numDimers"]=self.getNumDimers()
        return motionCounts


        motionCounts={}
        i=0
        while self.savedMotions[i]==None:
            i+=1
        for key in self.savedMotions[i].keys():
            motionCounts[key]=0

        motionCounts["numFrames"]=0
        motionCounts["complexMotionVertices"]=0

        for motionSet in self.savedMotions:
            if motionSet==None:
                continue

            motionCounts["numFrames"]+=1


            for type, motions in motionSet.items():
                if type=="complex":
                    enteredStrings=0
                    for motion in motions:
                        enteredStrings+=len(motion[0])
                        for string in motion[0]:
                            motionCounts["complexMotionVertices"]+=len(string.getVertexPoints(string.lattice))
                    motionCounts[type]+=enteredStrings
                else:
                    motionCounts[type]+=len(motions)
        

        motionCounts["numStrings"]=self.getNumStrings()
        motionCounts["numDimers"]=self.getNumDimers()
        return motionCounts


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

    print(motion.getMotionSummary())

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
            
