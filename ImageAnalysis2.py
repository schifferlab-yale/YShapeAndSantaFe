import argparse
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

#Get the image and make sure it exists
im=[];
try:
    im = cv2.imread(args.image[0])
    im = cv2.resize(im, (1000,1000))
except:
    raise Exception("File not found")

rawIm=im.copy()
imWidth, imHeight, imChannels = im.shape

#CONSTANTS
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(255,0,0)
RED=(0,0,255)
sampleWidth=10;

TOPLEFT=0
TOPRIGHT=1
CENTER=2
BOTTOM=3

#Global variables
outputImage=[];#image to draw the final output on
islands=[]#array of all the islands
avgColor=0;#Average color of the entire image

#These are the points used to generate the grid
#topLeft, topRight, bottomLeft, second, arm
#referencePoints=[[43,30],[974,37],[41, 962],[103, 34], [42,55]]
#referencePoints=[[10,10],[900,10],[10,900],[60,10],[20,20]]
referencePoints=[]

referenceLabels=["Top Left", "Top Right", "Bottom Left", "Second", "Arm"]

#Index of point currently being drug
selectedPoint=-1;
#Is the user dragging a point
dragging=False;
#Set this flag to true so the screen is redrawn if the user has finished dragging a point
hasUpdate=True;



class Node():
    def __init__(self, x,y, isBlack):
        self.x=x
        self.y=y
        self.isBlack=isBlack



#Fires every time mouse is moved or clicked on main window
def mouse_click(event, x, y,flags, param):
    global outputImage, islands, hasUpdate, sampleWidth
    global circles, dragging, selectedPoint

    #Begin dragging reverence point
    if event == cv2.EVENT_LBUTTONDOWN:
        if(len(referencePoints)==0):
            referencePoints.append([x,y])
            print("Please click on the center of the top right island")
        elif(len(referencePoints)==1):
            referencePoints.append([x,y])
            print("Please click on the center of the bottom left island")
        elif(len(referencePoints)==2):
            referencePoints.append([x,y])
            print("Please click on the center of the second island from the left in the top row")
        elif(len(referencePoints)==3):
            referencePoints.append([x,y])
            print("Please click on the bottom arm of the top left island")
        elif(len(referencePoints)==4):
            referencePoints.append([x,y])
        else:
            selectedPoint=getClosestReferencePoint(x,y)
            dragging=True;

    #Release dragging reference point
    elif event == cv2.EVENT_LBUTTONUP:
        dragging=False;
        hasUpdate=True;

    #Update reference point if dragging
    elif event == cv2.EVENT_MOUSEMOVE:
        if(dragging):
            referencePoints[selectedPoint]=[x,y]

    #Change the color of the closest node
    elif event == cv2.EVENT_RBUTTONDOWN:
        showScreen()

        #Find closest node
        closestRow=0;
        closestIsland=0;
        closestNode=0;
        minDist=100000;

        for (i, row) in enumerate(islands):
            for (j,island) in enumerate(row):
                for (k,node) in enumerate(island):
                    distance=dist([node[0],node[1]],[x,y])
                    if distance<minDist:
                        minDist=distance;
                        closestRow=i;
                        closestIsland=j;
                        closestNode=k;

        #swap color of closest node
        if(islands[closestRow][closestIsland][closestNode][2]==0):
            islands[closestRow][closestIsland][closestNode][2]=255
        else:
            islands[closestRow][closestIsland][closestNode][2]=0

    showScreen();


#Gets the color of an area on the screen at (x,y)
#It will look at every pixel within a certain with of the specified point and then
#average their values
def getColor(x,y):
    avg=0;
    count=0;#number of pixels checked
    width=sampleWidth;#distance away from center pixel to sample
    x=int(x)
    y=int(y)

    #loop through rows and columns
    for i in range(x-width,x+width):
        for j in range(y-width,y+width):
            #make sure pixel is not offscreen
            if i<0 or j<0 or i>imWidth-1 or j>imHeight-1:
                continue
            #add to average
            avg+=im[j][i][0]#(im[j][i][0]+im[j][i][1]+im[j][i][2])/3
            count+=1

    #prevent divide by 0 error
    if count==0:
        return 0

    #return avg color
    avg/=count;
    return avg

#return 0 or 255 based on whether area is black or white
def getType(x,y):
    if(getColor(x,y)<avgColor):
        return 255
    else:
        return 0

#Returns the average color of the entire image
def getAverageColor(im):
    avgColor=0;
    width, height, channels = im.shape
    for i in range(width):
        for j in range(height):
            avgColor+=(int(im[i][j][0])+int(im[i][j][1])+int(im[i][j][2]))/3
    avgColor/=width*height;
    return avgColor
avgColor=getAverageColor(im)

#distance between [x1,y1] and [x2,y2]
def dist(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2)+math.pow(point1[1] - point2[1], 2))

#get the closest reference point to a given (x,y)
def getClosestReferencePoint(x,y):
    minDist=100000;
    selected=0;
    for (i,point) in enumerate(referencePoints):
        distance=dist([x,y],point)
        if(distance<minDist):
            selected=i;
            minDist=distance;
    return selected


#Draw everything onto the screen
def showScreen():
    global outputImage, islands, hasUpdate, width, height

    #Reset the output image to be a blank grey image
    outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    outputImage[:,:]=(127,127,127)

    #reset the screen image to be the default image to be analyzed
    im=rawIm.copy();

    #draw the referencePoints
    for (label,point) in zip(referenceLabels, referencePoints):
        im=cv2.circle(im,(point[0],point[1]), sampleWidth, BLUE, 5)
        if dragging:
            im=cv2.putText(im, label, (point[0],point[1]+20),cv2.FONT_HERSHEY_SIMPLEX ,0.4,(0,0,255),1,cv2.LINE_AA)


    if(len(referencePoints)>=5):
        #Get the 4 reference points
        topLeft=referencePoints[0]
        topRight=referencePoints[1]
        bottomLeft=referencePoints[2]
        second=referencePoints[3]
        arm=referencePoints[4]

        hSpacing=dist(topLeft,second)#spacing between islands
        hCount=round(dist(topRight,topLeft)/hSpacing+1);#estimate the number of horizontal islands based on the spacing
        hSpacing=dist(topLeft,topRight)/hCount;#reestimate the spacing now that the number of islands are known

        vSpacing=hSpacing*math.sqrt(3)/2#vertical spacing is just a multiple of the horizontal spacing
        vCount=round(dist(topLeft, bottomLeft)/vSpacing+1);#estimate number of vertical islansd
        vSpacing=dist(topLeft,topRight)/vCount;#get a better value for vspacing now that vcount is known

        #distance and angle between center of island and lower "arm"
        armDist=dist(topLeft, arm)
        armAngle=math.atan2(topLeft[1]-bottomLeft[1],topLeft[0]-bottomLeft[0]);

        #calculate where the bottom right island is
        bottomRight=[int(bottomLeft[0]+hCount*hSpacing), int(bottomLeft[1]+(topRight[1]-topLeft[1]))]

        #Show guidelines when dragging reference points
        if dragging:
            #square box
            im=cv2.line(im,(topLeft[0], topLeft[1]),(topRight[0],topRight[1]),RED,2)
            im=cv2.line(im,(topLeft[0], topLeft[1]),(bottomLeft[0],bottomLeft[1]),RED,2)
            im=cv2.line(im,(topRight[0], topRight[1]),(bottomRight[0],bottomRight[1]),RED,2)
            im=cv2.line(im,(bottomLeft[0], bottomLeft[1]),(bottomRight[0],bottomRight[1]),RED,2)

            #3 "arms"
            im=cv2.line(im,(topLeft[0],topLeft[1]),(int(topLeft[0]-armDist*math.cos(armAngle)),int(topLeft[1]-armDist*math.sin(armAngle))),BLACK, 4)
            im=cv2.line(im,(topLeft[0],topLeft[1]),(int(topLeft[0]-armDist*math.cos(armAngle+2*math.pi/3)),int(topLeft[1]-armDist*math.sin(armAngle+2*math.pi/3))),BLACK, 4)
            im=cv2.line(im,(topLeft[0],topLeft[1]),(int(topLeft[0]-armDist*math.cos(armAngle+4*math.pi/3)),int(topLeft[1]-armDist*math.sin(armAngle+4*math.pi/3))),BLACK, 4)

            rowIsOdd=False;
            for (rowX, rowY) in zip(np.linspace(topLeft[0], bottomLeft[0],vCount),np.linspace(topLeft[1], bottomLeft[1],vCount)):
                #make offset overy other row
                if(rowIsOdd):
                    rowX+=hSpacing/2
                rowIsOdd=not rowIsOdd

                #loop through all cols in row
                for (colX, colY) in zip(np.linspace(rowX,rowX+hCount*hSpacing, hCount ) ,np.linspace(rowY, rowY+(topRight[1]-topLeft[1]),hCount)):
                    im=cv2.circle(im, (int(colX),int(colY)),2,RED,-1)

        else:
            #if the positions of the islands need to be recalculated
            if(hasUpdate):
                hasUpdate=False;

                #This alternates for every row to generate the offset
                rowIsOdd=False;

                islands=[]
                rowIndex=-1;

                #loop through all rows
                for (rowX, rowY) in zip(np.linspace(topLeft[0], bottomLeft[0],vCount),np.linspace(topLeft[1], bottomLeft[1],vCount)):
                    islands.append([])
                    rowIndex+=1;
                    #make offset overy other row
                    if(rowIsOdd):
                        rowX+=hSpacing/2
                    rowIsOdd=not rowIsOdd

                    #loop through all cols in row
                    for (colX, colY) in zip(np.linspace(rowX,rowX+hCount*hSpacing, hCount ) ,np.linspace(rowY, rowY+(topRight[1]-topLeft[1]),hCount)):
                        #generate location of 3 arms
                        bottomArm=[colX-armDist*math.cos(armAngle),colY-armDist*math.sin(armAngle)]
                        leftArm=[colX-armDist*math.cos(armAngle+2*math.pi/3),colY-armDist*math.sin(armAngle+2*math.pi/3)]
                        rightArm=[colX-armDist*math.cos(armAngle+4*math.pi/3),colY-armDist*math.sin(armAngle+4*math.pi/3)]

                        #append everything into array using format [x,y,color]
                        islands[rowIndex].append([  [leftArm[0],leftArm[1],getType(leftArm[0],leftArm[1])], [rightArm[0],rightArm[1],getType(rightArm[0],rightArm[1])], [colX,colY,getType(colX,colY)], [bottomArm[0],bottomArm[1],getType(bottomArm[0],bottomArm[1])]    ])

            for(i, row) in enumerate(islands):
                for (j,island) in enumerate(row):
                    #count the number of black and white to make sure there are two of each
                    black=0;
                    white=0;

                    centerX=int(island[CENTER][0])
                    centerY=int(island[CENTER][1])

                    #draw center of island
                    if(island[CENTER][2]==0):
                        white+=1
                        im=cv2.circle(im, (centerX,centerY),sampleWidth,WHITE,2)
                    else:
                        black+=1;
                        im=cv2.circle(im, (centerX,centerY),sampleWidth,BLACK,2)

                    #If it is the top left corner of a loop
                    color=island[TOPRIGHT][2]
                    if(len(row)>j+1 and len(islands)>i+1):
                        right=row[j+1];
                        if(i%2==0):
                            rightBelow=islands[i+1][j]
                        else:
                            rightBelow=islands[i+1][j+1]

                        if(right[BOTTOM][2]==color and right[TOPLEFT][2]!=color and rightBelow[TOPLEFT][2]==color and rightBelow[TOPRIGHT][2]!=color and island[BOTTOM][2]!=color):
                            drawX=(island[CENTER][0]+right[CENTER][0])/2
                            drawY=(2*island[CENTER][1]+rightBelow[CENTER][1])/3
                            if(color==0):
                                outputImage=cv2.circle(outputImage,(int(drawX),int(drawY)),20,(RED),3)
                            else:
                                outputImage=cv2.circle(outputImage,(int(drawX),int(drawY)),20,(BLUE),3)


                    #loop through each arm
                    for node in [island[0],island[1],island[3]]:
                        #skip off screen nodes
                        if(node[0]<0 or node[0]>imWidth or node[1]<0 or node[1]>imHeight):
                            continue
                        #draw
                        if node[2]==0:
                            outputImage=cv2.arrowedLine(outputImage, (int(node[0]),int(node[1])), (centerX,centerY), WHITE, 2, tipLength=0.2)
                            im=cv2.circle(im, (int(node[0]),int(node[1])),sampleWidth,WHITE,2)
                            white+=1;
                        else:
                            outputImage=cv2.arrowedLine(outputImage, (centerX,centerY), (int(node[0]),int(node[1])), BLACK, 2, tipLength=0.2)
                            im=cv2.circle(im, (int(node[0]),int(node[1])),sampleWidth,BLACK,2)
                            black+=1;

                    #draw a red circle if the number of white and black nodes is off
                    if(black>2 or white>2):
                        c=cv2.circle(im,(centerX,centerY), int(armDist*2), (0,0,255), 3)

    cv2.imshow("nopic",outputImage)
    cv2.imshow("window",im)



showScreen();
print("Please click on the center of the top left island")
cv2.setMouseCallback('window', mouse_click)

cv2.waitKey(0)
cv2.imwrite("output.jpg", np.float32(outputImage));

def islandsToString(islands):
    string=""
    for row in islands:
        for island in row:
            for arm in (island[TOPLEFT][2], island[TOPRIGHT][2], island[CENTER][2], island[BOTTOM][2]):
                if(arm==0):
                    string+="-1, ";
                else:
                    string+="1, ";
            string +="\t"
        string+="\n"
    return string

with open('output.csv', 'w') as file:
    file.write(islandsToString(islands))


# close all the opened windows.
cv2.destroyAllWindows()
