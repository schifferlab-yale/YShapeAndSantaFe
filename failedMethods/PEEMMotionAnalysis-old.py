import rotatePEEM
import santafeAnalysis
import argparse
import csv
import numpy as np
import cv2
import time

#given a set of strings, return the ones with unique IDs
def getUniqueStrings(strings):
    outStrings=[]
    outIDs=[]
    badIDs=[]#bad IDs keeps track of strings with the same ID (we don't want those)
    for string in strings:
        if(string.id not in badIDs):#if id is good
            if(string.id in outIDs):#if a string already has this id, remove both strings and mark id as bad
                outStrings.pop(outIDs.index(string.id))
                outIDs.remove(string.id)
                badIDs.append(string.id)
            else:
                outStrings.append(string)
                outIDs.append(string.id)
    
    return outStrings

#returns strings wihich combined
def getCombined(oldStrings,newStrings):

    combined={}

    for string in newStrings:
        #first make sure this string was new, not from previous
        valid=True
        for string2 in oldStrings:
            if(string.id==string2.id):
                valid=False
                break
        if not valid:
            continue

        #potential sources are strings whose centers are a subset of the larger strings centers
        potentialSources=[]
        for string2 in oldStrings:
            if(string.id==string2.id):
                continue
            couldBeSource=True

            #all points in string must also be contained in bigger string
            for point in string2.touchedInteriors:
                if point not in string.touchedInteriors:
                    couldBeSource=False
                    break

            #can't be a source if that string still exists
            for newString in newStrings:
                if(newString.id==string2.id):
                    couldBeSource=False
                    break
            
            if(couldBeSource):
                potentialSources.append(string2)
        
        #loop through all combinations of potential sources to see if any of them form the bigger string
        if(len(potentialSources)>=2):
            for source1 in potentialSources:
                for source2 in potentialSources:
                    #cant be a combination if they are the same string or their 
                    if(source1==source2 or source1.id==string.id or source2.id==string.id):
                        continue
                    interiors1=set(source1.touchedInteriors)
                    interiors2=set(source2.touchedInteriors)
                    if interiors1 | interiors2==set(string.touchedInteriors) and not interiors1.issubset(interiors2) and not interiors2.issubset(interiors1):
                        combined[string.id]=[source1.id,source2.id]
    return combined

def getLengthChanges(oldStrings,newStrings):
    grew, shrunk, sameLength = [], [], []

    oldStrings=getUniqueStrings(oldStrings)
    newStrings=getUniqueStrings(newStrings)

    newStringIDs=[string.id for string in newStrings]

    for string in oldStrings:
        if string.id in newStringIDs:
            oldLength=string.getLength();
            newLength=newStrings[newStringIDs.index(string.id)].getLength()
            if(oldLength<newLength):
                grew.append(string)
            elif(oldLength>newLength):
                shrunk.append(string)
            else:
                sameLength.append(string)

    return grew,shrunk,sameLength



def getLoopCreations(oldStrings,newStrings):

    oldIDs=[string.id for string in oldStrings if string.isLoop()]
    newIDs=[string.id for string in newStrings if string.isLoop()]
    newStrings=[string for string in newStrings if string.isLoop()]


    created=[]

    for string, newID in zip(newStrings, newIDs):
        if newID not in oldIDs :
            created.append(string)
    
    return created

def getWiggles(oldStrings,newStrings):
    wiggled=[]

    newStrings=getUniqueStrings(newStrings)
    oldStrings=getUniqueStrings(oldStrings)

    newStringIDs=[string.id for string in newStrings]

    for string in oldStrings:
        if(string.id in newStringIDs):
            otherString=newStrings[newStringIDs.index(string.id)]
            if( set(string.getPoints()) != set(otherString.getPoints()) and string.getLength()==otherString.getLength()):
                wiggled.append((string,otherString))
    
    return wiggled


if __name__ == "__main__":
    start=time.time()

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

    lastStrings={}
    outImages=[]

    for islandIndex,islands in enumerate(islandData[0:99]):
        #turn PEEM data into a usable format for the santafelattice class
        rotated=rotatePEEM.rotatePEEM(rowData,colData,islands)
        rotated=rotated.split("\n")

        #create lattice
        lattice=santafeAnalysis.SantaFeLattice(rotated, removeEdgeStrings=False, autoAlignCenters=True, randomizeStringColor=True)
        #lattice.removeSmallStrings()
        #lattice.removeStringsNotConnectingInteriors()
        strings=lattice.strings

        #draw output
        outputImage=np.zeros((1000,1000,3), np.uint8)
        outputImage[:,:]=(250,250,250)
        lattice.drawStrings(outputImage,lineWidth=3,showID=True)
        lattice.drawCells(outputImage)


        #use above functions to get string changes
        grew, shrunk, sameLength = getLengthChanges(lastStrings,strings)
        combined=getCombined(lastStrings,strings)
        split=getCombined(strings,lastStrings)
        createdLoops=getLoopCreations(lastStrings, strings)
        annihilatedLoops=getLoopCreations(strings, lastStrings)
        wiggled=getWiggles(lastStrings, strings)


        #write changes onto image
        outString=""# "grew:"+str(len(grew))+", shrunk:"+str(len(shrunk))+", split:"+str(len(split))+", combined:"+str(len(combined))+"\n\n"
        for string in grew:
            outString+="string "+str(string.id)+" grew\n"

        for string in shrunk:
            outString+="string "+str(string.id)+" shrunk\n"


        for splitString, splitTo in split.items():
            outString+="string "+str(splitString)+" split into "+str(splitTo[0])+" and "+str(splitTo[1])+"\n"

        for combinedString, combinedFrom in combined.items():
            outString+="strings "+str(combinedFrom[0])+" and "+str(combinedFrom[1])+" combined into "+str(combinedString)+"\n"

        for string in createdLoops:
            outString+="Loop "+str(string.id)+" was created\n"

        for string in annihilatedLoops:
            outString+="Loop "+str(string.id)+" was annihilated\n"
        
        for stringPair in wiggled:
            outString+="string "+str(stringPair[0].id)+" wiggled\n"

        for i, outLine in enumerate(("Changes from previous:\n"+outString).split("\n")):
            cv2.putText(outputImage, outLine, (10,700+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)
        
        if(len(outImages)>0):
            for i, outLine in enumerate(("Changes to next:\n"+outString).split("\n")):
                cv2.putText(outImages[len(outImages)-1], outLine, (700,700+i*15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0,0,0), 1, cv2.LINE_AA)



        #lattice.drawStringTraces(outputImage)
        
        lastStrings=strings
        #cv2.imshow("window",outputImage)
        outImages.append(outputImage)
        #cv2.waitKey(0)

    for i, image in enumerate(outImages):
        cv2.imwrite("out/"+str(i)+".jpg", np.float32(image))
    
    print(time.time()-start)
        
