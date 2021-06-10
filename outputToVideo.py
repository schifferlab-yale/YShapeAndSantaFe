import os, argparse
import subprocess


if __name__ == "__main__":
    #get file from args
    parser = argparse.ArgumentParser(description='directory to convert (recursively)')
    parser.add_argument("directory", type=str, help="Path to directory")
    args=parser.parse_args()

    mainDir=args.directory

    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, mainDir)
    
    directoryToCreate = os.path.join(mainDir,"videos")
    try:
        os.mkdir(directoryToCreate)
    except FileExistsError:
        pass

    for currentpath, folders, files in os.walk(directory):
        if(len(files)>0):
            try:
                os.chdir(currentpath)
                fileName=currentpath.split(mainDir)[1].replace("\\","-").replace("/","-").replace(":","-").replace(" ","-").replace(".csv","")
                #print(fileName)
                outputFile=dirname+"\\"+os.path.join(mainDir,"videos")+"\\"+fileName+".mp4"
                #print(outputFile)
                cmd="ffmpeg -r 2 -f image2  -i %d.jpg -vcodec libx264 -crf 30  -pix_fmt yuv420p "+outputFile+" -loglevel quiet"
                print(cmd)
                os.system(cmd)
            except Exception as e:
                print(e)