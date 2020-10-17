import cv2
import numpy as np
import utils


########################################################################
webCamFeed = False
pathImage = "Images/tomato.jpg"
# cap = cv2.VideoCapture(0)
# cap.set(10,160)
heightImg = 1000
widthImg  = 700
########################################################################

#utils.initializeTrackbars()
count=0
thresh_ct = 0
x = cv2.imread(pathImage)

while True:
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1) # ADD GAUSSIAN BLUR
    thres = [240 - thresh_ct, 240 - thresh_ct] #utils.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=4) # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utils.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0 or thresh_ct == 200:
        break
    thresh_ct += 10

if biggest.size != 0:
    biggest=utils.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    imgBigContour = utils.drawRectangle(imgBigContour,biggest,2)
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    #REMOVE 20 PIXELS FORM EACH SIDE
    imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

    # APPLY ADAPTIVE THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre,3)
    imgAdaptiveDial = cv2.dilate(imgAdaptiveThre, (3,3), iterations=4)
    imgAdaptiveEro = cv2.erode(imgAdaptiveDial, kernel, iterations=4)


    # Image Array for Display
    #imageArray = ([img,imgGray,imgThreshold,imgContours],
                 # [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

# else:
#     imageArray = ([img,imgGray,imgThreshold,imgContours],
#                   [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    #lables = [["Original","Gray","Threshold","Contours"],
            #["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    #stackedImage = utils.stackImages(imageArray,0.75,lables)
    #cv2.imshow("Result",imgAdaptiveThre)

    # SAVE IMAGE WHEN 's' key is pressed

#cv2.imshow("a",imgGray)
#cv2.imshow("as",imgBlur)
cv2.imshow("threshold",imgThreshold)
cv2.imshow("contours",imgContours)
cv2.imshow("gray",imgWarpGray)
cv2.imshow("adaptive thresh",imgAdaptiveThre)
cv2.imshow("colored",imgWarpColored)
cv2.imshow("adaptive erosion",imgAdaptiveEro)
cv2.waitKey(0)
cv2.destroyAllWindows()


# while 1:
#     cv2.imshow(":)",imgContours)
#     if cv2.waitKey(1) & 0xFF == ord('s'):
#         cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
#         cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
#                       (1100, 350), (0, 255, 0), cv2.FILLED)
#         cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
#                     cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
#         cv2.imshow('Result', stackedImage)
#         cv2.waitKey(300)
#         count += 1
#     if 0xFF == ord('q'):
#         cv2.destroyWindow(":)")
#         break
