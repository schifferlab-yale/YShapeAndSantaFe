from newPeemMotionAnalysis import analyzeFile, determineFileType, vertexEnergies
import glob, os, cv2
import numpy as np



if __name__=="__main__":
    folder="cleanedPEEMFiles"

    curDir=os.path.dirname(__file__)
    files = glob.glob(curDir + '/'+folder+'/**/*.csv', recursive=True)


    madeHeader=False
    motionKeys=[]

    with open(curDir+"/out/out.txt","a") as outFile:
        for (fileI, file) in enumerate(files):
            print(f"({fileI+1}/{len(files)}) {file}")

            relativePath=file.split(folder)[1]

        
            try:
                type=determineFileType(relativePath)
                energies=vertexEnergies[type]
            except Exception:
                energies=None
                print("could not find energies for "+fileI)
            
            

            try:
                motionCounts,outImages,_ = analyzeFile(file,skippedIndex=10)


                #TEXT
                if(not madeHeader):
                    madeHeader=True
                    
                    motionKeys=motionCounts.keys()
                    outFile.write("file name, ")
                    for key in motionKeys:
                        outFile.write(key+", ")
                    outFile.write("\n")
                
                outFile.write(file.split(folder)[1]+", ")
                for key in motionKeys:
                    outFile.write(str(motionCounts[key])+", ")
                outFile.write("\n")

                #IMAGES
                
                outDir=curDir+"/out"+relativePath+"/"

                os.makedirs(outDir, exist_ok=True)
                for i,file in enumerate(outImages):
                    savePath=outDir+str(i)+".jpg"
                    success=cv2.imwrite(savePath, np.float32(file))
                    if not success:
                        raise Exception("could not write image")
            except Exception as e:
                raise
                print(e)