
# Author: Shreyash Annapureddy
# Date: 16/09/2015
# Modifications: Implementing the optmiser to solve for depth of object


import numpy as np
import cv
import cv2
import roslib
import rospy
import sensor_msgs
import time
import cv_bridge
import scipy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import baxter_interface
from scipy.optimize import minimize 

global vertexOrdinates
global webcam
global baxterFocal
global webMx
global webMy
global baxterMx
global baxterMy
global targetEdge
global targetDiagonal
global numItems
global numItemsJac

numItems = 0
numItemsJac = 0



# Create a dictonary of edges and diagonal distances of known models
targetEdge = {'Square':50.0/1000.0,'Octogon':22.5/1000.0}
targetDiagonal = {'Square':2*(50.0/1000.0)**2,'Octogon':(58.80/1000.0)**2}

# Determining the conversion factor between pixel to real world distances
baxterFocal = 1.2
baxterMx = 409.60391/baxterFocal
baxterMy = 408.97915/baxterFocal






class Baxter(object):
    def __init__(self,limb):
        self._arm = limb
        self._limb = baxter_interface.Limb(self._arm)
        self._rs = baxter_interface.RobotEnable()
        self._init_state = self._rs.state().enabled
        self._rs.enable()


    def Clean_shutdown(self):
        self._rs.disable()

    def Left_cam_open(self):
        cam_left = baxter_interface.camera.CameraController("left_hand_camera")
        cam_left.close()
        cam_left.open()
        cam_left.resolution = (640,400)

    def Left_cam_close(self):
        cam_left = baxter_interface.camera.CameraController("left_hand_camera")
        cam_left.close()

    def cam_head_open(self):
        cam_head = baxter_interface.camera.CameraController("head_camera")
        cam_head.close()
        cam_head.open()

    def cam_head_close(self):
        cam_head = baxter_interface.camera.CameraController("head_camera")
        cam_head.close()
    
    def Disp_head(self,img):
        self.pub = rospy.Publisher('/robot/xdisplay', img, latch = True)

def _identifyShapes(hierarchy, contours):
	# max number of objects that can be detected (subject to change)
    maxNumObjects = 200
	# max area of enclosed contour, if the dimensions of the contour are less that 30px x 30px
	# Most likely noise
    maxArea = 30*30
	# Hirarchy in effect contains how many countours have been detected (size is nx4)
    numObjects = np.size(hierarchy)/4
    #print numObjects
	# Check if the the number of objects do not exceed limit
    if numObjects < maxNumObjects:
		# Going through every detected object to find moment, centroid and area
        for index in range(0,numObjects):
            area = cv2.contourArea(contours[index])
            if area > maxArea:
                epsilon = 0.02*cv2.arcLength(contours[index], True)
                approx = cv2.approxPolyDP(contours[index], epsilon, True)
                approx = modVertexFilter(approx)
                numVertices = len(approx)
                #print numVertices
                moment = cv2.moments(contours[index])
                centroidX = int(moment['m10']/moment['m00'])
                centroidY = int(moment['m01']/moment['m00'])
                if numVertices == 3:
                    shape = 'Triangle'
                elif numVertices == 4:
                    shape = 'Square'
                elif numVertices == 5:
                    shape = 'Pentagon'
                elif numVertices == 6:
                    shape = 'Hexagon'
                elif numVertices == 7:
                    shape = 'Heptagon'
                elif numVertices == 8:
                    shape = 'Octogon'
                else:
                    shape = 'No shape indentified'
                center = [centroidX,centroidY,shape]
                return (center, approx)
    center = [0,0,'']
    return (center, [])



def _removeNoise(obj):
    newFrame = cv2.cvtColor(obj, cv2.COLOR_HSV2BGR)
    imGray = cv2.cvtColor(newFrame,cv2.COLOR_BGR2GRAY)
    laplaceKernal = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
    imGray = LaplacianSharpening(imGray,laplaceKernal)
    thresh = cv2.adaptiveThreshold(imGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,np.ones((5,5),'uint8'))
    thresh_inv = cv2.bitwise_not(thresh)
    return (thresh_inv,imGray)

def findShape(frame, color,	chemistsquareWidth = 30):
    global vertexOrdinates
    global baxterMx
    global baxterMy
    cntX = 0
    cntY = 1
    shape = 2

	# Convert Colors to numpy array
    lowColor = np.array(color['low'])
    highColor = np.array(color['high'])

	# Mask the frame
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    colorMask = cv2.inRange(hsvFrame, lowColor, highColor)
    objectMask = cv2.bitwise_and(frame, frame, mask=colorMask)
	
	
    thresh_inv, imGray = _removeNoise(objectMask)

    contours, hierarchy = cv2.findContours(thresh_inv,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    center, vertexOrdinates = _identifyShapes(hierarchy, contours)
    tempVertexOrdinates = np.copy(vertexOrdinates)
    if center[shape] != '':
        for index in range(len(tempVertexOrdinates)):
            coordinates = tempVertexOrdinates[index]
            cv2.circle(frame, (coordinates.item((0,0)),coordinates.item((0,1))), 5,(0,0,255), -1)
    cv2.circle(frame, (center[cntX],center[cntY]), 3,(0,0,0), -1)
    cv2.putText(frame, center[shape], (center[cntX]+60,center[cntY]),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(0,0,255))
    return (frame,center[shape])




def modVertexFilter(vertexOrdinates):
    threshold = 28.36
    toBeAvg = []
    finalPoints = []
    for items in range(0,len(vertexOrdinates)):
        if items == len(vertexOrdinates)-1:
            ordinate1 = vertexOrdinates[items]
            ordinate2 = vertexOrdinates[0]
        else:
            ordinate1 = vertexOrdinates[items]
            ordinate2 = vertexOrdinates[items+1]
            dist = ((ordinate2[0][0] - ordinate1[0][0])**2 + (ordinate2[0][1] - ordinate1[0][1])**2)**0.5
        if dist < threshold:
            toBeAvg.append(ordinate1)
        else:
            toBeAvg.append(ordinate1)
            vertex = np.average(toBeAvg, axis = 0)
            vertex = np.round(vertex)
            vertex = vertex.astype(dtype = 'uint32')
            finalPoints.append(vertex)
            toBeAvg = []
    return finalPoints

def SaturationCorrection(frame, color):
	if color == 'green':
		hue_change = 65
	HSV_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	H,S,V = cv2.split(HSV_img)
	S = S.astype(float)
	S_copy = np.copy(S)
	min_S = np.amin(S)
	max_S = np.amax(S)
	epsilon_S = (min_S + max_S)/3.0
	b = 2
	kernalConst = 1.0/((2.0+b)**2.0)
	kernalVect = np.array([[1.0],[b],[1.0]])
	kernal = (np.dot(kernalVect,kernalVect.T))*kernalConst
	S_filtered = cv2.filter2D(S_copy,-1,kernal)
	Coordiantes = np.where(S_filtered < epsilon_S)
	rows = Coordiantes[0]
	cols = Coordiantes[1]
	for i in range(0,len(rows)):
		S_filtered[rows[i]][cols[i]] = 0
		V[rows[i]][cols[i]] = 0
	Coordiantes = np.where(S_filtered >= epsilon_S)
	rows = Coordiantes[0]
	cols = Coordiantes[1]
	for i in range(0,len(rows)):
		H[rows[i]][cols[i]] = hue_change
		S_filtered[rows[i]][cols[i]] = 250
		V[rows[i]][cols[i]] = 180
	S_filtered = S_filtered.astype(dtype = 'uint8')
	correctedImg = cv2.merge((H,S_filtered,V))
	return correctedImg

def LaplacianSharpening(frame,kernal):
    originalImg = np.copy(frame)
    originalImg = originalImg.astype(float)
    filtered = cv2.filter2D(originalImg,-1,kernal)
    filtered = filtered - np.amin(filtered)
    filtered = filtered * (255.0/np.amax(filtered))
    sharpenedImg = frame + filtered
    sharpenedImg = sharpenedImg - np.amin(sharpenedImg)
    sharpenedImg = sharpenedImg * (255.0/np.amax(sharpenedImg))
    sharpenedImg = sharpenedImg.astype(dtype = 'uint8')
    return sharpenedImg

def gammaCorrection(frame,correction):
	channel_0,channel_1,channel_2 = cv2.split(frame)
	channel_0 = channel_0/255.0
	channel_0 = channel_0**correction
	channel_0 = np.uint8(channel_0*255)

	channel_1 = channel_1/255.0
	channel_1 = channel_1**correction
	channel_1 = np.uint8(channel_1*255)

	channel_2 = channel_2/255.0
	channel_2 = channel_2**correction
	channel_2 = np.uint8(channel_2*255)

	gamImg = cv2.merge((channel_0,channel_1,channel_2))
	return gamImg


def OptCenterTransform(pixelPos):
    principlePoint = np.array([[308.94946, 225.93870]])
    Ox = principlePoint[0][0]
    Oy = principlePoint[0][1]
    xOriginal = pixelPos[0][0]
    yOriginal = pixelPos[0][1]
    uOriginal = xOriginal - Ox
    vOriginal = yOriginal - Oy
    pixelPos[0][0] = uOriginal
    pixelPos[0][1] = vOriginal
    return pixelPos
    

def objective(t,shape):
    global targetEdge
    global targetDiagonal
    global vertexOrdinates
    global baxterFocal
    global baxterMx
    global baxterMy
    vertexCopy = np.copy(vertexOrdinates)
    v = ()
    for index in range(len(vertexOrdinates)):
        ordinate = vertexCopy[index]
        ordinate = ordinate.astype(dtype = 'float32')
        ordinate = OptCenterTransform(ordinate)
        ordinate[0][0] = ordinate[0][0]/baxterMx
        ordinate[0][1] = ordinate[0][1]/baxterMy
        ordinate = np.append(ordinate,baxterFocal)
        # Converting the distances from mm to meters
        ordinate = 1e-3*ordinate
        v += (ordinate,)
    # Lambda function to iterate over each evertex for edge and diagonal distances
    next = lambda k: (k+1)%len(vertexCopy)
    diag = lambda k: (k+(len(vertexCopy)/2))%len(vertexCopy)

    # Sum the edge distance with respect to a target, this acts as the 1st half of the cost function
    sideError = [(np.linalg.norm(v[i]*t[i]-v[next(i)]*[next(i)])**2 - targetEdge[shape]**2)**2 for i in range(len(vertexCopy))]
    sideSum = sum(sideError)

    # Sum the diagonal distance with respect to a target, this act as the second half of the cost function
    diagonalError = [(np.linalg.norm(t[i]*v[i]-t[diag(i)]*v[diag(i)])**2 - targetDiagonal[shape])**2 for i in range(len(vertexCopy)-3)]
    diagonalSum = sum(diagonalError)
    # Return a regularised error
    return 1000000*(diagonalSum+sideSum)


def t_initGuess():
    global vertexOrdinates
    t = np.ones(len(vertexOrdinates))
    return t*25.0

def convert2mm():
    global vertexOrdinates
    for index in range(len(vertexOrdinates)):
        coordinates = vertexOrdinates[index]
        coordinates = coordinates.astype(dtype = 'float32')
        coordinatesFloat = OptCenterTransform(coordinates)
        coordinatesFloat[0][0] = coordinates[0][0]/baxterMx
        coordinatesFloat[0][1] = coordinates[0][1]/baxterMy
        vertexOrdinates[index] = coordinatesFloat

def determineConstraint(t):
    global vertexOrdinates
    global numItems
    if numItems == len(vertexOrdinates):
        numItems = 0
    function = np.array([t[numItems]])
    numItems += 1
    return function

def determineJacConstraint(t):
    global vertexOrdinates
    global numItemsJac
    if numItemsJac == len(vertexOrdinates):
        numItemsJac = 0
    function = np.zeros(len(vertexOrdinates))
    function[numItemsJac] = 1.0
    numItemsJac += 1
    return function






def constraint(cons,t):
    global vertexOrdinates
    for index in range(len(vertexOrdinates)):
        items = {}
        items['type'] = 'ineq'
        items['fun'] = lambda t: determineConstraint(t)
        items['jac'] = lambda t: determineJacConstraint(t)
        cons += (items,)
    return cons




def minimisation(shape,t0):
    global vertexOrdinates
    cons = constraint((),t0)
    res = minimize(objective,t0,args = (shape),method = 'SLSQP',constraints = cons ,options = {'disp':True})
    print "-----------------------------------------"
    print res.x
    #convert2mm()
    #print vertexOrdinates
    #print "-----------------------------------------"
    return res
    



def main(contourWindow = True, grayscaleWindow = True, binaryWindow = True):
    global vertexOrdinates
    shapeChange = 0
    rospy.init_node('Detect_object', anonymous = True)
    blue = {'low': [95,100,100], 'high': [125,227,227]}
    green = {'low': [50,100,100], 'high': [70,255,255]}
    left = Baxter('left')
    left.Left_cam_open()
    while(True):
        img_ROS = rospy.wait_for_message('/cameras/left_hand_camera/image',Image)
        img_cv = CvBridge().imgmsg_to_cv(img_ROS, "bgr8")
        frame = np.asarray(img_cv)
        pic_gamCorrect = gammaCorrection(frame,0.6)
        pic_satCorrect = SaturationCorrection(pic_gamCorrect,'green')
        pic_final = cv2.cvtColor(pic_satCorrect,cv2.COLOR_HSV2BGR)
        greenFrame,shape = findShape(pic_final, green)
        if shapeChange - len(vertexOrdinates) != 0:
            t0 = t_initGuess()
            shapeChange = len(vertexOrdinates)
        if shape == 'Square' or shape == 'Octogon':
            res = minimisation(shape,t0)
            t0 = res.x
        if contourWindow:
            cv2.imshow('Contours', greenFrame)
            cv2.imshow('saturated img', pic_satCorrect)
        if cv2.waitKey(1) == 1048689:
            left.Left_cam_close()
            #left.cam_head_close()
            break
    cv2.destroyAllWindows()
    left.Clean_shutdown()

if __name__ == "__main__":
    main()



