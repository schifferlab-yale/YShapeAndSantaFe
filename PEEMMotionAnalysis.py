import rotatePEEM
import santafeAnalysis
import argparse
import csv
import numpy as np
import cv2

def getNoChange(oldStrings,newStrings):
    noChange=[]
    for oldString in oldStrings:
        for newString in newStrings:
            if set(oldString.getPoints()) == set(newString.getPoints()):
                noChange.append((oldString,newString))
    
    return noChange

#Assume that duplicate strings have already been removed
def getWiggleGrowShrink(oldStrings,oldLattice,newStrings,newLattice):
    wiggle,grow,shrink=[],[],[]


    pairs = getStringPairs(oldStrings, oldLattice, newStrings, newLattice)

    for pair in pairs:
        if(pair[0].getSegmentCount() == pair[1].getSegmentCount()):
            wiggle.append(pair)
        elif(pair[0].getSegmentCount() > pair[1].getSegmentCount()):
            shrink.append(pair)
        else:
            grow.append(pair)
    

    return wiggle,grow,shrink



def groupByTouchingSquares(oldStrings,oldLattice,newStrings,newLattice):

    groups={}

    for newString in newStrings:
        
        touchingCenters=frozenset(newLattice.getTouchingCompositeSquares(newString))
        hashKey=hash(touchingCenters)
        if hashKey not in groups:
            groups[hashKey]=([],[newString])

            for string in newStrings:
                thisHash=hash(frozenset(newLattice.getTouchingCompositeSquares(string)))
                if thisHash == hashKey and string != newString:
                    groups[hashKey][1].append(string)
            
            for string in oldStrings:
                thisHash=hash(frozenset(oldLattice.getTouchingCompositeSquares(string)))
                if thisHash == hashKey:
                    groups[hashKey][0].append(string)
    
    return list(groups.values())

def matchupBySharedPoints(strings1, strings2):
    #the principle for this is that we will loop through the shorter list and find each elements best pair from teh longer list

    #make copies of lists since we will be deleting elements from them
    strings1 = [string for string in strings1]
    strings2 = [string for string in strings2]

    
    #make sure strings1 is the shorter of the two lists
    newFirst=True
    if len(strings1)>len(strings2):
        strings1, strings2 = strings2, strings1
        newFirst=False

    #now sort by number of points, descending. This is becaues we want to check longer strings first because they will be more likel yto share points
    strings1 = sorted(strings1, key = lambda el: len(el.getPoints()), reverse=True)


    pairs=[]

    for string1 in strings1:
        string1Points=set(string1.getPoints())

        bestCorrelation=-1
        bestMatch=strings2[0]

        #find string which pairs best with it
        for string2 in strings2:
            correlation=len(string1Points.intersection(set(string2.getPoints())))
            if(correlation>bestCorrelation):
                bestMatch=string2
                bestCorrelation=correlation
        
        if(newFirst):
            pairs.append((string1,bestMatch))
        else:
            pairs.append((bestMatch,string1))
        strings2.remove(bestMatch)#remove selected string so that it onlyy gets mapped to once

    return pairs


def getStringPairs(oldStrings,oldLattice,newStrings,newLattice):
    pairs=[]
    groups=groupByTouchingSquares(oldStrings, oldLattice, newStrings, newLattice)

    for group in groups:

        #if there is no corresponding string between frames
        if len(group[0]) == 0 or len(group[1])==0:
            continue
        #if the strings match up perfectly
        elif len(group[0])==1 and len(group[1])==1:
            pairs.append((group[0][0],group[1][0]))
        #else things don't line up
        else:
            pairs+=matchupBySharedPoints(group[0],group[1])    

    return pairs    

def getMerges(oldStrings,oldLattice,newStrings,newLattice):
    merges=[]

    for potentialMerged in newStrings:
        mergedPoints=set(potentialMerged.getPoints())

        bestCoverage=-1
        sources=[]

        for string1 in oldStrings:
            string1Points=set(string1.getPoints())
            string1CoveredPoints=mergedPoints.intersection(string1Points)
            if len(string1CoveredPoints)==0:
                continue
            for string2 in oldStrings:
                if string1==string2:
                    continue
                string2Points=set(string2.getPoints())
                string2CoveredPoints=mergedPoints.intersection(string2Points)
                if(len(string2CoveredPoints)==0):
                    continue
                totalCoveredPoints=len(string1CoveredPoints.union(string2CoveredPoints))
                if(totalCoveredPoints>bestCoverage):
                    bestCoverage=totalCoveredPoints
                    sources=[string1,string2]
        
        if(bestCoverage>1):
            merges.append((sources,potentialMerged))





    #merge based off interior centers, not very good
    """for potentialMerged in newStrings:
        touchingSquares=newLattice.getTouchingCompositeSquares(potentialMerged)

        #a merged string must touch at least 2 squares
        if len(touchingSquares)<2:
            continue

        sources=[]

        for potentialSource in oldStrings:
            isSource=False
            for point in oldLattice.getTouchingCompositeSquares(potentialSource):
                if point in touchingSquares:
                    isSource=True
                    break
            
            if(isSource):
                sources.append(potentialSource)
        
        if(len(sources)>=2):
            merges.append((sources,potentialMerged))"""


    return merges

def getLoopWiggleExpandContract(oldStrings,oldLattice,newStrings,newLattice):
    oldLoops = [string for string in oldStrings if string.isLoop()]
    newLoops = [string for string in newStrings if string.isLoop()]

    pairs = matchupBySharedPoints(oldLoops, newLoops)

    wiggle, expand, contract = [], [], []

    for pair in pairs:
        
        if(pair[0].getSegmentCount() < pair[1].getSegmentCount()):
            expand.append(pair)
        elif(pair[0].getSegmentCount() > pair[1].getSegmentCount()):
            contract.append(pair)
        else:
            wiggle.append(pair)
    

    return wiggle, expand, contract



def latticeComparison(oldLattice,newLattice):
    oldStrings=[string for string in oldLattice.strings]
    newStrings=[string for string in newLattice.strings]


    #NO CHANGE *******************************************
    noChange=getNoChange(oldStrings,newStrings)
    
    for oldString, newString in noChange:
        oldStrings.remove(oldString)
        newStrings.remove(newString)

    #WIGGLE GROW SHRINK **************************
    wiggle,grow,shrink=getWiggleGrowShrink(oldStrings,oldLattice,newStrings,newLattice)

    for stringType in [wiggle,grow,shrink]:
        for oldString, newString in stringType:
            oldStrings.remove(oldString)
            newStrings.remove(newString)
    
    #MERGE SPLIT ******************************************

    merge=getMerges(oldStrings,oldLattice,newStrings,newLattice)
    split=getMerges(newStrings,newLattice,oldStrings,oldLattice)
    for merged in merge:
        for source in merged[0]:
            if source in oldStrings:
                oldStrings.remove(source)
        newStrings.remove(merged[1])
    
    for splitStrings in split:
        for source in splitStrings[0]:
            if source in newStrings:
                newStrings.remove(source)
        if splitStrings[1] in oldStrings:
            oldStrings.remove(splitStrings[1])
    
    
    ##LOOPS
    loopWiggle, loopExpand, loopContract = getLoopWiggleExpandContract(oldStrings,oldLattice,newStrings,newLattice)

    for stringType in [loopWiggle,loopExpand,loopContract]:
        for oldString, newString in stringType:
            oldStrings.remove(oldString)
            newStrings.remove(newString)
    
    #Creation/annihilation

    loopCreation=[]
    loopAnnihilation=[]
    stringCreation=[]
    stringAnnihilation=[]

    for string in oldStrings:
        if string.isLoop():
            loopAnnihilation.append(string)
        else:
            stringAnnihilation.append(string)
    for string in newStrings:
        if string.isLoop():
            loopCreation.append(string)
        else:
            stringCreation.append(string)

    


    return {"noChange":noChange, "wiggle":wiggle, "grow":grow, "shrink":shrink, "merge":merge,"split":split, "loopWiggle":loopWiggle, "loopExpand":loopExpand, "loopContract":loopContract, "loopCreation":loopCreation, "loopAnnihilation":loopAnnihilation, "stringCreation":stringCreation, "stringAnnihilation":stringAnnihilation}

def stringifyChanges(changes):
    text="Changes from last slide:\n"
    #for noChange in changes["noChange"]:
    #    text+=str(noChange[0].id)+" no change\n"
    text+=str(len(changes["noChange"]))+" strings: no change\n"

    for shrink in changes["shrink"]:
        text+=str(shrink[0])+"->"+str(shrink[1])+" shrunk\n"

    for shrink in changes["wiggle"]:
        text+=str(shrink[0])+"->"+str(shrink[1])+" wiggle\n"

    for shrink in changes["grow"]:
        text+=str(shrink[0])+"->"+str(shrink[1])+" grow\n"

    for merge in changes["merge"]:
        text+=str(merge[0])+" merged into "+str(merge[1])+"\n"

    for split in changes["split"]:
        text+=str(split[1])+" split into "+str(split[0])+"\n"

    for wiggle in changes["loopWiggle"]:
        text+=str(wiggle[0])+"->"+str(wiggle[1])+" loop wiggle\n"

    for expand in changes["loopExpand"]:
        text+=str(expand[0])+"->"+str(expand[1])+" loop expand\n"

    for contract in changes["loopContract"]:
        text+=str(contract[0])+"->"+str(contract[1])+" loop contract\n"
    
    for string in changes["loopCreation"]:
        text+="loop "+str(string)+" created\n"
    
    for string in changes["loopAnnihilation"]:
        text+="loop "+str(string)+" annihilated\n"

    for string in changes["stringCreation"]:
        text+=str(string)+" created\n"
    
    for string in changes["stringAnnihilation"]:
        text+=str(string)+" annihilated\n"
    
    
    return text

if __name__ == "__main__":
    #get file from args
    parser = argparse.ArgumentParser(description='Peem analysis')
    parser.add_argument("file", type=str, help="Path to csv file")
    args=parser.parse_args()

    #open file
    try:
        file=open(args.file,newline="\n")
    except:
        raise Exception("Error with file")

    with open(args.file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data=[row for row in data if len(row)>1]#remove empty rows
    data=[[int(float(cell)) for cell in row] for row in data]#convert to int

    #turn columns into rows
    rowData=[row[0] for row in data]
    colData=[row[1] for row in data]
    islandData=[[row[i] for row in data] for i in range(2,len(data[0]))]

    lastLattice=None
    outImages=[]

    #loop through
    for islandIndex,islands in enumerate(islandData[0:99]):
        print(f"frame {islandIndex}")
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=santafeAnalysis.SantaFeLattice(rotated, removeEdgeStrings=False, autoAlignCenters=True)

        #draw output
        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        lattice.drawStrings(outputImage,lineWidth=3,showID=True)
        lattice.drawCells(outputImage)


        

        outImages.append(outputImage)

        if lastLattice is not None:
            changes=latticeComparison(lastLattice,lattice)

            for i, outLine in enumerate((stringifyChanges(changes)).split("\n")):
                cv2.putText(outputImage, outLine, (10,700+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)

        lastLattice=lattice
    
    #draw images
    for i, image in enumerate(outImages):
        cv2.imwrite("out/"+str(i)+".jpg", np.float32(image))
    
