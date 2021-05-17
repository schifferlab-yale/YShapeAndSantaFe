from PEEMMotionAnalysis import analyzeFile
import glob, os, cv2
import numpy as np

if __name__=="__main__":
    curDir=os.path.dirname(__file__)
    files = glob.glob(curDir + '/PEEMFiles/**/*.csv', recursive=True)

    motionKeys=[]

    with open(curDir+"/out/out.txt","a") as outFile:
        for (fileI, file) in enumerate(files):
            print(f"({fileI+1}/{len(files)}) {file}")

            relativePath=file.split("PEEMFiles")[1]


            motionCounts,outImages = analyzeFile(file)


            #TEXT
            if(fileI==0):
                motionKeys=motionCounts.keys()
                outFile.write("file name, ")
                for key in motionKeys:
                    outFile.write(key+", ")
                outFile.write("\n")
            
            outFile.write(file.split("PEEMFiles\\")[1]+", ")
            for key in motionKeys:
                outFile.write(str(motionCounts[key])+", ")
            outFile.write("\n")

            #IMAGES
            
            outDir=curDir+"\out"+relativePath+"/"

            os.makedirs(outDir, exist_ok=True)
            
            for i,file in enumerate(outImages):
                cv2.imwrite(outDir+str(i)+".jpg", np.float32(file))