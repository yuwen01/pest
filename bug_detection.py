import cv2
import numpy as np
import utils

# start with colored image
pathImage = "stickytraps/1008.jpg"
heightImg = 800
widthImg  = 600

#img = cv2.imread(pathImage)
img = cv2.imread(pathImage, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (widthImg, heightImg))
imgBlur = cv2.GaussianBlur(img, (7, 7), 1)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 60
params.maxArea = 200

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.0001

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

#detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(imgBlur)
imgKeyPoints = cv2.drawKeypoints(imgBlur, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display found keypoints
cv2.imshow("Keypoints", imgKeyPoints)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
