import os, argparse

outFolder = "convertOut"
def getUserInputFiles():
    global outFolder

    parser = argparse.ArgumentParser(description='Y shaped lattice csv reader')
    parser.add_argument("file", type=str, help="Path to csv file or folder")
    parser.add_argument("--out", type=str, default="convertOut")
    args=parser.parse_args()

    outFolder=args.out

    fileNames=[]
    if os.path.isfile(args.file):
        fileNames=[args.file]
    elif os.path.isdir(args.file):
        for (dirpath, dirnames, files) in os.walk(args.file):
            for name in files:
                fileNames.append(os.path.join(dirpath,name))

    else:
        raise Exception("path is not file or directory")

    return fileNames

def convertFile(name):
    with open(name,"r") as f:
        text=f.read()
        text="first row offset\ntopLeft, topRight, middle, bottom\n"+text
        text=text.replace(",",", ")
        text=text.replace(".0","")

    return text

if __name__=="__main__":
    files=getUserInputFiles()

    

    for fileName in files:
        text=convertFile(fileName)

        outFileName=os.path.join(outFolder,fileName.split("/")[-1].split(".")[0][0:])+".csv"
        print(outFileName)
        with open(outFileName,"w") as outFile:
            outFile.write(text)