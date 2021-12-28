import rotatePEEM
import santafeAnalysis
import argparse
import csv
import numpy as np
import cv2
import os
import unittest
import json

#returns pairs of strings which connect exactly the same sets of points
def getNoChange(oldStrings,newStrings):
    noChange=[]
    for oldString in oldStrings:
        for newString in newStrings:
            if set(oldString.getPoints()) == set(newString.getPoints()):
                noChange.append((oldString,newString))

    return noChange

#pair up strings and return by size change
def getWiggleGrowShrink(oldStrings,oldLattice,newStrings,newLattice):
    wiggle,grow,shrink=[],[],[]


    pairs = getStringPairs(oldStrings, oldLattice, newStrings, newLattice)#pair up strings

    for pair in pairs:
        if(pair[0].getSegmentCount() == pair[1].getSegmentCount()):
            wiggle.append(pair)
        elif(pair[0].getSegmentCount() > pair[1].getSegmentCount()):
            shrink.append(pair)
        else:
            grow.append(pair)


    return wiggle,grow,shrink


#returns a list of tuples of lists where each list contains strings that touch the exact same points
#output [([oldStrings that touch these points],[newStrings that touch these points]), ([],[])]
def groupByTouchingSquares(oldStrings,oldLattice,newStrings,newLattice):

    #dictionary where key is the hash of the touching centers
    groups={}

    for mainString in newStrings+oldStrings:

        touchingCenters=frozenset(newLattice.getTouchingCompositeSquares(mainString))
        hashKey=hash(touchingCenters)

        if hashKey not in groups:#if this is a new hash, add to dict and check if there are others with this hash
            groups[hashKey]=([],[])

            for string in newStrings:
                thisHash=hash(frozenset(newLattice.getTouchingCompositeSquares(string)))
                if thisHash == hashKey:
                    groups[hashKey][1].append(string)

            for string in oldStrings:
                thisHash=hash(frozenset(oldLattice.getTouchingCompositeSquares(string)))
                if thisHash == hashKey:
                    groups[hashKey][0].append(string)

    return list(groups.values())

#tries to match up strings by number of shared points (algorithm is not perfect)
def matchupBySharedPoints(strings1, strings2, minShared=1):
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
            if(correlation>bestCorrelation and correlation>=minShared):
                bestMatch=string2
                bestCorrelation=correlation

        if(bestCorrelation>=minShared):
            if(newFirst):
                pairs.append((string1,bestMatch))
            else:
                pairs.append((bestMatch,string1))
            strings2.remove(bestMatch)#remove selected string so that it onlyy gets mapped to once

    return pairs

#pairs of strings into groups of 2 based on number of shared paints
def getStringPairs(oldStrings,oldLattice,newStrings,newLattice):
    pairs=[]
    groups=groupByTouchingSquares(oldStrings, oldLattice, newStrings, newLattice)
    ungrouped=([],[])

    for group in groups:

        #if there is no corresponding string between frames
        if len(group[0]) == 0 :
            ungrouped[1].extend(group[1 ])
        elif len(group[1])==0:
            ungrouped[0].extend(group[0])
        #if the strings match up perfectly
        elif len(group[0])==1 and len(group[1])==1:
            pairs.append((group[0][0],group[1][0]))
        #else things don't line up
        else:
            pairs+=matchupBySharedPoints(group[0],group[1])
    pairs+=matchupBySharedPoints(*ungrouped,minShared=2)
    return pairs


#if a new string shares at least 2 points with at least two old strings
def getMerges(oldStrings,oldLattice,newStrings,newLattice):
    merges=[]

    #loop through each new string and see if it can be made from two previous strings
    for potentialMerged in newStrings:
        mergedPoints=set(potentialMerged.getPoints())

        #old method that would only use two strings for a merge
        """
        bestCoverage=-1 #number of shared points from two other strings
        sources=[]

        for string1 in oldStrings:
            string1Points=set(string1.getPoints())
            string1CoveredPoints=mergedPoints.intersection(string1Points)
            if len(string1CoveredPoints)<2:#must have at least one shared point
                continue
            for string2 in oldStrings:
                if string1==string2:#cannot merge a string with itself!
                    continue
                string2Points=set(string2.getPoints())
                string2CoveredPoints=mergedPoints.intersection(string2Points)
                if(len(string2CoveredPoints)<2):#must cover at least one point
                    continue
                totalCoveredPoints=len(string1CoveredPoints.union(string2CoveredPoints))
                if(totalCoveredPoints>bestCoverage):
                    bestCoverage=totalCoveredPoints
                    sources=[string1,string2]

        if(bestCoverage>1):#this will always be true if we found a matching pair
            merges.append((sources,potentialMerged))
        """

        sources=[]
        for string in oldStrings:
            points=set(string.getPoints())
            if len(points.intersection(mergedPoints))>=2:
                sources.append(string)

        if(len(sources)>=2):
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

#matchup loops and return based on segment count
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

#get all the strings in the second argument which share at least n points with the first string
def getStringsSharingPoint(string,otherStrings,minShared=1):
    out=[]
    for otherString in otherStrings:
        sharedPoints=set(string.getPoints()).intersection(set(otherString.getPoints()))
        if(len(sharedPoints)>=minShared):
            out.append(otherString)
    return out


#if two strings in the old lattice share at least two seperate points with strings in the new lattice and do not touch

def getReconnection(oldStrings,oldLattice,newStrings,newLattice):
    oldStrings=[string for string in oldStrings]
    newStrings=[string for string in newStrings]

    reconnections=[]
    for oldString1 in oldStrings:
        potentialReconnects1=getStringsSharingPoint(oldString1,newStrings)
        for oldString2 in oldStrings:
            if(oldString1==oldString2):
                continue

            potentialReconnects2=getStringsSharingPoint(oldString2,newStrings)

            #now we have a set of new strings which share a point with two of the old strings
            potentialReconnects=set(potentialReconnects1).intersection(potentialReconnects2)

            #make sure that the point that they share is NOT the same

            for potentialReconnect in list(potentialReconnects):
                sharedPoints1=set(potentialReconnect.getPoints()).intersection(set(oldString1.getPoints()))
                sharedPoints2=set(potentialReconnect.getPoints()).intersection(set(oldString2.getPoints()))

                if len(sharedPoints1.union(sharedPoints2))<2:
                    potentialReconnects.remove(potentialReconnect)


            validReconnect=True

            #make sure the new strings do not touch
            for reconnect1 in potentialReconnects:
                for reconnect2 in potentialReconnects:
                    if(reconnect1==reconnect2):
                        continue
                    if len(set(reconnect1.getPoints()).intersection(set(reconnect2.getPoints())))!=0:
                        validReconnect=False

            #make sure we have at least two strings as a result

            if(len(potentialReconnects)<2):
                validReconnect=False



            if(validReconnect):
                reconnections.append(([oldString1,oldString2],list(potentialReconnects)))
                oldStrings.remove(oldString1)
                oldStrings.remove(oldString2)
                newStrings=[string for string in newStrings if string not in potentialReconnects]

                break #break out of the second string loop since the first string has already been classified




    return reconnections

def latticeComparison(oldLattice,newLattice,minEnergyDiff=1E-20):
    oldStrings=[string for string in oldLattice.strings]
    newStrings=[string for string in newLattice.strings]


    #NO CHANGE *******************************************
    noChange=getNoChange(oldStrings,newStrings)

    for oldString, newString in noChange:
        oldStrings.remove(oldString)
        newStrings.remove(newString)

    #RECONNECTION*********************************

    reconnection=getReconnection(oldStrings,oldLattice,newStrings,newLattice)

    for pair in reconnection:
        for oldString in pair[0]:
            oldStrings.remove(oldString)
        for newString in pair[1]:
            newStrings.remove(newString)

    #MERGE SPLIT ******************************************

    merge=getMerges(oldStrings,oldLattice,newStrings,newLattice)
    split=getMerges(newStrings,newLattice,oldStrings,oldLattice)#splits are just merges in the reverse direction
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

    #WIGGLE GROW SHRINK **************************
    wiggle,grow,shrink=getWiggleGrowShrink(oldStrings,oldLattice,newStrings,newLattice)

    for stringType in [wiggle,grow,shrink]:
        for oldString, newString in stringType:
            oldStrings.remove(oldString)
            newStrings.remove(newString)

    wiggleEnergyUnknownChange,wiggleEnergyIncrease,wiggleEnergyDecrease,wiggleNoEnergyChange=[],[],[],[]


    if(oldLattice.vertexEnergies is None):
        wiggleEnergyUnknownChange=wiggle
        wiggleNoEnergyChange=[]
        wiggleEnergyDecrease=[]
        wiggleEnergyIncrease=[]
    else:
        for pair in wiggle:
            change=newLattice.getStringEnergy(pair[1])-oldLattice.getStringEnergy(pair[0])

            if abs(change)<minEnergyDiff:
                wiggleNoEnergyChange.append(pair)
            elif change<0:
                wiggleEnergyDecrease.append(pair)
            elif change>0:
                wiggleEnergyIncrease.append(pair)
            else:
                print("should never happen")
                wiggleEnergyUnknownChange.append(pair)




    #if one is in a split and a merge, it is a reconnection

    #reconnection=[]
    """for merged in merge:
        for splitStrings in split:
            if merged[1] in splitStrings[0] and merged[1] not in reconnection:
                reconnection.append(merged[1])
                #merge.remove(merged)
                #split.remove(splitStrings)"""

    #remove reconnections from split and merge
    merge=[merged for merged in merge if merged[1] not in reconnection]

    for string in reconnection:
        for splitString in split:
            if string in splitString[0]:
                split.remove(splitString)




    ##LOOPS ********************************************
    loopWiggle, loopExpand, loopContract = getLoopWiggleExpandContract(oldStrings,oldLattice,newStrings,newLattice)

    for stringType in [loopWiggle,loopExpand,loopContract]:
        for oldString, newString in stringType:
            oldStrings.remove(oldString)
            newStrings.remove(newString)

    #Creation/annihilation (everything left over)

    loopCreation=[]
    loopAnnihilation=[]
    centerStringCreation=[]
    centerStringAnnihilation=[]
    edgeStringCreation=[]
    edgeStringAnnihilation=[]

    for string in oldStrings:
        if string.isLoop():
            loopAnnihilation.append(string)
        elif len(string.getPoints())==3 and len(oldLattice.getTouchingInteriors(string))==2:
            centerStringAnnihilation.append(string)
        else:
            edgeStringAnnihilation.append(string)


    for string in newStrings:
        if string.isLoop():
            loopCreation.append(string)
        elif len(string.getPoints())==3 and len(newLattice.getTouchingInteriors(string))==2:
            centerStringCreation.append(string)
        else:
            edgeStringCreation.append(string)




    return {
        "noChange":noChange,
        "wiggleEnergyUnknownChange":wiggleEnergyUnknownChange,
        "wiggleEnergyIncrease":wiggleEnergyIncrease,
        "wiggleEnergyDecrease":wiggleEnergyDecrease,
        "wiggleNoEnergyChange":wiggleNoEnergyChange,
        "grow":grow,
        "shrink":shrink,
        "merge":merge,
        "split":split,
        "reconnection":reconnection,
        "loopWiggle":loopWiggle,
        "loopExpand":loopExpand,
        "loopContract":loopContract,
        "loopCreation":loopCreation,
        "loopAnnihilation":loopAnnihilation,
        "centerStringCreation":centerStringCreation,
        "centerStringAnnihilation":centerStringAnnihilation,
        "edgeStringCreation":edgeStringCreation,
        "edgeStringAnnihilation":edgeStringAnnihilation,
        }

def stringifyChanges(changes,oldLattice,newLattice):
    text="Changes from last slide:\n"
    #for noChange in changes["noChange"]:
    #    text+=str(noChange[0].id)+" no change\n"
    text+=str(len(changes["noChange"]))+" strings: no change\n"

    for shrink in changes["shrink"]:
        text+=str(shrink[0])+"->"+str(shrink[1])+" shrunk\n"

    for wiggle in changes["wiggleEnergyUnknownChange"]:
        text+=str(wiggle[0])+"->"+str(wiggle[1])+" wiggle (unknown energy change)"
        text+="\n"

    for wiggle in changes["wiggleEnergyIncrease"]:
        text+=str(wiggle[0])+"->"+str(wiggle[1])+" wiggle (energy increase)"
        text+="\n"

    for wiggle in changes["wiggleEnergyDecrease"]:
        text+=str(wiggle[0])+"->"+str(wiggle[1])+" wiggle (energy decrease)"
        text+="\n"

    for wiggle in changes["wiggleNoEnergyChange"]:
        text+=str(wiggle[0])+"->"+str(wiggle[1])+" wiggle (no energy change)"
        text+="\n"



    for grow in changes["grow"]:
        text+=str(grow[0])+"->"+str(grow[1])+" grow\n"

    for merge in changes["merge"]:
        text+=str(merge[0])+" merged into "+str(merge[1])+"\n"

    for split in changes["split"]:
        text+=str(split[1])+" split into "+str(split[0])+"\n"

    for reconnection in changes["reconnection"]:
        text+=str(reconnection)+" reconnection \n"

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

    for string in changes["edgeStringCreation"]:
        text+="edge "+str(string)+" created\n"

    for string in changes["edgeStringAnnihilation"]:
        text+="edge "+str(string)+" annihilated\n"

    for string in changes["centerStringCreation"]:
        text+="center "+str(string)+" created\n"

    for string in changes["centerStringAnnihilation"]:
        text+="center "+str(string)+" annihilated\n"


    return text


def analyzeFile(fileName,minFrame=0,maxFrame=-1,debug=False,vertexEnergies=None,label=""):
     #open file
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

    lastLattice=None


    outImages=[]
    allMotions=[]
    motionCounts=None

    if(maxFrame==-1):
        maxFrame=len(islandData)
    #loop through
    for islandIndex,islands in enumerate(islandData[minFrame:maxFrame]):
        if(debug):
            print(f"frame {islandIndex}")
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=santafeAnalysis.SantaFeLattice(rotated, removeEdgeStrings=False, autoAlignCenters=True, vertexEnergies=vertexEnergies)

        #make sure it is not blank
        if(lattice.isBlank()):
            continue

        #draw output
        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        lattice.drawStrings(outputImage,lineWidth=3,showID=True)
        lattice.drawCells(outputImage,flagCell = lambda row,col: lastLattice is not None and lattice.getCell(row,col).arrow != lastLattice.getCell(row,col).arrow)

        blank=np.zeros((1000,500,3), np.uint8)
        blank[:,:]=(250,250,250)
        outputImage=np.concatenate((outputImage,blank), axis=1)

        if lastLattice is not None:
            changes=latticeComparison(lastLattice,lattice)

            if(motionCounts is None):
                motionCounts={}
                for key in changes.keys():
                    motionCounts[key]=0

            for key,val in changes.items():
                motionCounts[key]+=len(val)

            allMotions.append(changes)


            #this splits it into lines
            lineLength=60
            lines=(stringifyChanges(changes,lastLattice,lattice)).split("\n")
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
                cv2.putText(outputImage, outLine, (1000,15+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)

            cv2.putText(outputImage,label,(0,15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(outputImage,"frame "+str(islandIndex),(0,30), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)


            pastImage=np.zeros((1000,1000,3), np.uint8)
            pastImage[:,:]=(250,250,250)
            lastLattice.drawStrings(pastImage,lineWidth=10,showID=False)
            lastLattice.drawCells(pastImage)
            pastImage=cv2.cvtColor(pastImage, cv2.COLOR_BGR2GRAY)
            _,pastImage=cv2.threshold(pastImage, 240, 255, cv2.THRESH_BINARY)
            pastImage=cv2.cvtColor(pastImage, cv2.COLOR_GRAY2BGR)
            pastImage=np.concatenate((pastImage,blank), axis=1)

            outputImage=cv2.addWeighted(outputImage,0.8,pastImage,0.2,0)

        outImages.append(outputImage)

        lastLattice=lattice

    return motionCounts, outImages, allMotions




if __name__ == "__main__":
    #get file from args
    parser = argparse.ArgumentParser(description='Peem analysis')
    parser.add_argument("file", type=str, help="Path to csv file")
    parser.add_argument('-n', "--numImages", metavar='n', help="number of images to analyze", type=int, default=-1)
    args=parser.parse_args()

    if args.file=="test":
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        exit();

    motionCounts, outImages, allMotions = analyzeFile(args.file,maxFrame=args.numImages)

    dirname = os.path.dirname(__file__)


    directoryToCreate = os.path.join(dirname, "out",args.file.split("\\")[-1])
    try:
        os.mkdir(directoryToCreate)
    except FileExistsError:
        pass

    #draw images
    for i, image in enumerate(outImages):
        cv2.imwrite("out/"+args.file.split("\\")[-1]+"/"+str(i)+".jpg", np.float32(image))


    with open("out/out.txt","a") as file:

       file.write("\n"+args.file+"\n")
       motionTotal=0;
       for key, value in motionCounts.items():
            if key != "noChange":
                motionTotal+=value
       for key,value in motionCounts.items():
            if key == "noChange":
                file.write(str(key)+": "+str(value)+"\n")
            else:
                file.write(str(key)+": "+str(value)+"/"+str(motionTotal)+"\n")










