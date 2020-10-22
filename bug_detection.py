import cv2
import numpy as np
import utils

# start with colored image
pathImage = "stickytraps/1170.jpg"
widthImg = 600
heightImg = 800

#Gray scale and blur the image
img = cv2.imread(pathImage)
imgGray = cv2.imread(pathImage, cv2.IMREAD_GRAYSCALE)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
height, width, channels = img.shape

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 400
#params.maxArea = 3000

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.00001

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.000001


# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

#detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(imgBlur)
print(len(keypoints))
imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Crop out keypoints
str = ""
for keypoint in keypoints:
    str += "1"
    x = int(keypoint.pt[0])
    y = int(keypoint.pt[1])
    size = int(keypoint.size)
    cv2.imshow(str, img[max(1,y-2*size): min(height-1,y+2*size), max(1,x-2*size): min(width-1,x+2*size)])


# Display found keypoints
imgKeyPoints = cv2.resize(imgKeyPoints, (2000, 1400))
cv2.imshow("Keypoints", imgKeyPoints)
#cv2.imshow("grays", imgGray)
#cv2.imshow("blur", imgBlur)
#cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
