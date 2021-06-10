from PEEMMotionAnalysis import analyzeFile
import glob, os, cv2
import numpy as np


#Example energy dict:
"""
{
    2 (island vertex):{
        (type) 1: energy
        (type) 2: energy
    }
    3 (island vertex):{
        (type) 1: energy
        (type) 2: energy
        (type) 3: energy
    }
    4 (island vertex):{
        (type) 1: energy
        (type) 2: energy
        (type) 3: energy
        (type) 4: energy
    }
}
"""
vertexEnergies={}
#Sept 2016------------------------------------
vertexEnergies["Sep_2016_600"]={
    2:{
        1:1.421E-18,
        2:1.532E-18
    },
    3:{
        1:2.025E-18,
        2:2.083E-18,
        3:2.422E-18
    },
    4:{
        1:2.640E-18,
        2:2.693E-18,
        3:2.865E-18,
        4:3.482E-18
    }
},
vertexEnergies["Sep_2016_700"]={
    2:{
        1:1.565E-18,
        2:1.644E-18
    },
    3:{
        1:2.268E-18,
        2:2.312E-18,
        3:2.553E-18
    },
    4:{
        1:2.991E-18,
        2:3.030E-18,
        3:3.149E-18,
        4:3.583E-18
    }
}
vertexEnergies["Sep_2016_800"]={
    2:{
        1:,
        2:
    },
    3:{
        1:,
        2:,
        3:
    },
    4:{
        1:,
        2:,
        3:,
        4:
    }
}

#May 2017------------------------------------
vertexEnergies["May_2017_700"]={
    2:{
        1:1.450E-18,
        2:1.512E-18
    },
    3:{
        1:2.123E-18,
        2:2.164E-18,
        3:2.348E-18
    },
    4:{
        1:2.792E-18,
        2:2.847E-18,
        3:2.930E-18,
        4:3.254E-18
    }
}
vertexEnergies["May_2017_800"]={
    2:{
        1:1.489E-18,
        2:1.530E-18
    },
    3:{
        1:2.201E-18,
        2:2.231E-18,
        3:2.353E-18
    },
    4:{
        1:2.905E-18,
        2:2.953E-18,
        3:3.004E-18,
        4:3.213E-18
    }
}
#Mar 2018------------------------------------
vertexEnergies["Mar_2018_600"]={
    2:{
        1:1.081E-18,
        2:1.135E-18
    },
    3:{
        1:1.586E-18,
        2:1.623E-18,
        3:1.780E-18
    },
    4:{
        1:2.073E-18,
        2:2.135E-18,
        3:2.199E-18,
        4:2.467E-18
    }
}
#Nov 2019------------------------------------
vertexEnergies["Nov_2019_600"]={
    2:{
        1:1.076E-18,
        2:1.121E-18	
    },
    3:{
        1:1.583E-18,	
        2:1.615E-18,	
        3:1.747E-18	
    },
    4:{
        1:2.075E-18,	
        2:2.130E-18,	
        3:2.184E-18,	
        4:2.408E-18
    }
}
vertexEnergies["Nov_2019_700"]={
    2:{
        1:1.218E-18,
        2:1.255E-18	
    },
    3:{
        1:1.800E-18,	
        2:1.827E-18,	
        3:1.937E-18	
    },
    4:{
        1:2.372E-18,	
        2:2.416E-18,	
        3:2.462E-18,	
        4:2.650E-18
    }
}
vertexEnergies["Nov_2019_800"]={
    2:{
        1:1.343E-18,	
        2:1.374E-18	
    },
    3:{
        1:1.991E-18,	
        2:2.014E-18,	
        3:2.107E-18	
    },
    4:{
        1:2.631E-18,	
        2:2.670E-18,	
        3:2.708E-18,	
        4:2.866E-18
    }
}

#Nov 2020------------------------------------
vertexEnergies["Nov_2020_600"]={
    2:{
        1:1.422E-18,
        2:1.545E-18	
    },
    3:{
        1:1.998E-18,	
        2:2.061E-18,	
        3:2.443E-18	
    },
    4:{
        1:2.615E-18,	
        2:2.648E-18,	
        3:2.831E-18,	
        4:3.407E-18
    }
}
vertexEnergies["Nov_2020_700"]={
    2:{
        1:1.414E-18,	
        2:1.470E-18	
    },
    3:{
        1:2.067E-18,	
        2:2.100E-18,	
        3:2.270E-18	
    },
    4:{
        1:2.731E-18,	
        2:2.767E-18,	
        3:2.848E-18,	
        4:3.148E-18
    }
}
vertexEnergies["Nov_2020_800"]={
    2:{
        1:1.386E-18,
        2:1.417E-18	
    },
    3:{
        1:2.047E-18,	
        2:2.067E-18,	
        3:2.163E-18	
    },
    4:{
        1:2.717E-18,	
        2:2.739E-18,	
        3:2.789E-18,	
        4:2.953E-18
    }
}


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

if __name__=="__main__":
    folder="cleanedPEEMFiles"

    curDir=os.path.dirname(__file__)
    files = glob.glob(curDir + '/'+folder+'/**/*.csv', recursive=True)

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
                print("could not find energies for"+fileI)
            
            

            try:
                motionCounts,outImages,_ = analyzeFile(file,debug=False,vertexEnergies=energies,label=relativePath)


                #TEXT
                if(fileI==0):
                    motionKeys=motionCounts.keys()
                    outFile.write("file name, ")
                    for key in motionKeys:
                        outFile.write(key+", ")
                    outFile.write("\n")
                
                outFile.write(file.split(folder+"\\")[1]+", ")
                for key in motionKeys:
                    outFile.write(str(motionCounts[key])+", ")
                outFile.write("\n")

                #IMAGES
                
                outDir=curDir+"\out"+relativePath+"/"

                os.makedirs(outDir, exist_ok=True)
                for i,file in enumerate(outImages):
                    cv2.imwrite(outDir+str(i)+".jpg", np.float32(file))
            except Exception as e:
                #raise
                print(e)