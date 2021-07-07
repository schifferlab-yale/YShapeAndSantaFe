import cv2 #for drawing images
import numpy as np #cv2 likes numpy arrays  
import argparse #to get the image from the user
from nodeNetwork import * #Use this as the base class 

#Set up argparser to allow for input image
parser = argparse.ArgumentParser(description='Perpendicular Kagome MFM image analysis')
parser.add_argument('image', metavar='image', type=str, nargs='+',help='Path of image')
args=parser.parse_args()

#How big the display window(s) are
WINDOWSIZE=500

#read image
image = cv2.imread(args.image[0])
if image is None:
    raise FileNotFoundError("Cannot find image")

#resize image (makes it easier to work with)
image=cv2.resize(image,(WINDOWSIZE,WINDOWSIZE))

#this is the pattern of where to sample the image at
#the squares with "1" in them, will sample at that point, while the ones with "0" will not
#make sure each row has the same number of elements in it
gridPattern=[
    [0,0,1,0],
    [0,1,0,1],
    [1,0,0,0],
    [0,1,0,1]
]


#rowOffset and colOffset allow you to shift the grid around (so you're not always starting in the exact same place in the pattern)
rowOffset=0
colOffset=0

class PerpendicularKagomeReader(NodeNetwork):

    #given a row,column, and the coordinants of four corners of a square in the grid, where should the sample points in that square be?
    #output format is [[x1,y1],[x2,y2],[x3,y3]]...
    #in this case, we only have one or zero sample points in a square so the output will be either [[x,y]] or []
    def getSamplePointsFromSquare(self, topLeft, topRight, bottomLeft, bottomRight, row=0,col=0):
        
        #find out whiche spot in the gridPattern this square corresponds to based on its row,col and our offsets
        patternRow=(row+rowOffset)%len(gridPattern)
        patternCol=(col+colOffset)%len(gridPattern[0])

        #we only sample that point if the value in the gridPattern at that point is 1
        if gridPattern[patternRow][patternCol]==1:

            #We simply want the center of the square, so we just average the four points and return
            centerX=(topLeft[0]+topRight[0]+bottomLeft[0]+bottomRight[0])/4
            centerY=(topLeft[1]+topRight[1]+bottomLeft[1]+bottomRight[1])/4
            return [[centerX,centerY]]
        else:
            #if this is a 0 in the gridpattern, there is no sample point in this square
            return []

    #as far as I can tell, there is no way to detect errors in this lattice type (no impossible color/charge configurations)
    def hasError(self, samplePoints, rowI, vertexI, pointI):
        return False
    
    #this is how we convert the data to the output format for the file
    def dataAsString(self):
        string=""#the string that will get put into the file

        #loop through each row and square in that row
        for (rowI, row) in enumerate(self.samplePoints):
            for (vertexI, vertex) in enumerate(row):

                #in theory, a vertex is an array of points, so we might have to loop through it and dump those points into the file
                #however, in the case of this grid there will only ever be 1 or 0 points in a vertex, so we can just look at vertex[0]


                if len(vertex)>0:
                    #the color of the point is stored at index 2 (sorry this is kind of arbitrary)
                    string+=str(vertex[0][2])+", "
                else:
                    #placeholder if there is nothing
                    string+=" ,"
            string+="\n"
        return string
    

#make the grid overlay by giving it the four corners, the number of rows and columns, the image we want to sample, and how big the sample area should be
#when it determines the color of a point in an image, it doesn't just look at that one point, it averages in the area around it. This is what pointSampleRadius is for.
#pointSampleRadius should be roughly half the width of a single dot
n=PerpendicularKagomeReader(Node(10,10),Node(WINDOWSIZE-10,10),Node(10,WINDOWSIZE-10),Node(WINDOWSIZE-10,WINDOWSIZE-10),15,15,image,pointSampleRadius=5)

def show():

    #this first ouput draws the grid and sample points over our main image
    outputImage=image.copy()
    n.draw(outputImage)
    cv2.imshow("window",outputImage)

    #this is the preview image
    imWidth=WINDOWSIZE;
    imHeight=WINDOWSIZE;
    outputImage=np.zeros((imHeight,imWidth,3), np.uint8)
    outputImage[:,:]=(127,127,127)
    n.drawData(outputImage)
    cv2.imshow("output",outputImage)



#this controls what mouse events do
def mouse_event(event, x, y,flags, param):

    #split the grid when on right click
    if event == cv2.EVENT_RBUTTONDOWN:
        n.splitAtClosestPoint(x,y)
    
    
    elif event ==cv2.EVENT_LBUTTONDOWN:

        #manually toggle the value of a sample point with shift+left click
        if(flags==16 or flags==17):
            n.toggleNearestSamplePoint(x,y)
        
        #on normal left click, start dragging
        else:
            n.selectNearestFixedPoint(x,y)
            n.dragging=True

    #update dragging on mouse move
    elif event==cv2.EVENT_MOUSEMOVE:
        n.updateDragging(x,y)

    #release dragging on mouse up
    elif event==cv2.EVENT_LBUTTONUP:
        n.stopDragging()

    #show the update image every frame
    show()

show()
#Bind the mouse event function to the window
cv2.setMouseCallback('window', mouse_event)


print("Controls")
print("Right click: split the grid at that point")
print("Left click + drag: drag around a reference point")
print("Shift + Left Click: manually correct the value of a point")
print("r/e: add or remove a row")
print("c/x: add or remove a column")
print("o/p: adjust the row/column offset")

while(True):
    #wait for the next key press
    key=cv2.waitKey(0)

    #end the program on "Enter"
    if(key==ord("\r")):
        break;
    
    #add/remove row with r/e
    elif(key==ord("r")):
        n.addRow()
    elif(key==ord("e")):
        n.removeRow()

    #add/remove column with c/x
    elif(key==ord("c")):
        n.addCol()
    elif(key==ord("x")):
        n.removeCol()

    #use o and p to offset the row and column
    elif(key==ord("o")):
        rowOffset=(rowOffset+1)%len(gridPattern)
        n.setSamplePoints()
    elif(key==ord("p")):
        colOffset=(colOffset+1)%len(gridPattern)
        n.setSamplePoints()


    show()

#get the name of the output file based on the input (same name but change extension to .csv)
csvName=args.image[0].split(".")[0]+".csv"
print("writing to... "+csvName)
with open(csvName, 'w') as file:#write filedata
    file.write(n.dataAsString())

#write outputImage
outputImage=np.zeros((WINDOWSIZE,WINDOWSIZE,3), np.uint8)
outputImage[:,:]=(127,127,127)
n.drawData(outputImage)
cv2.imwrite("output.jpg", np.float32(outputImage));

#close the windows
cv2.destroyAllWindows()