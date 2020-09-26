# Perspective of live streaming of paper as whiteboard
# Paper Detection
import cv2
import numpy as np

video = cv2.VideoCapture(1)  # WebCam
frameWidth = 640
frameHeight = 700

video.set(3, frameWidth)
video.set(4, frameHeight)
video.set(100, 150)

store_coordinate_points = []

# Function for preprocessing
def preProcessing(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gray image
    imgBlur = cv2.GaussianBlur(imgGray, (1, 1), 1)  # Blur Image
    imgCanny = cv2.Canny(imgBlur, 100, 300)  # Canny Image
    kernel = np.ones((5, 5))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDilate, kernel, iterations=1)

    return imgErode

# Function to get coordinate points of the paper
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
       RETR_EXTERNAL -> store the first external edge
       CHAIN_APPROX_NONE -> store all the boundary points
    '''
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Calculates area of each shape detected
        if area > 5000:
            peri = cv2.arcLength(cnt, True)  # arcLength -> perimeter of the closed shape
            cornerPoints = cv2.approxPolyDP(cnt, 0.01*peri, True)
            if area > maxArea and len(cornerPoints) == 4:
                biggest = cornerPoints
                maxArea = area

    return biggest


# Function to get warp perspective of the paper
def getWarp(image, page_coord_points):
    # store_coordinate_points stores the previous coordinates points of the page. It is used when there is any object
    # intervention in between live stream
    global store_coordinate_points

    if not coord_points.any():
        # print(" Outside object intervention")
        prev_coord_points = store_coordinate_points[:]
        pts1 = np.float32([prev_coord_points[1], prev_coord_points[0], prev_coord_points[2], prev_coord_points[3]])
        pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, matrix, (frameWidth, frameHeight))
    else:
        # Reshaping the array from (4, 1, 2) to (4, 2)
        new_coord_points = np.reshape(page_coord_points, (4, 2))

        # Assign the obtained page corner points to the global variable
        store_coordinate_points = new_coord_points[:]

        # Coordinates of the book edges
        pts1 = np.float32([new_coord_points[1], new_coord_points[0], new_coord_points[2], new_coord_points[3]])
        pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, matrix, (frameWidth, frameHeight))

    return image


while True:
    success, img = video.read()
    # img = cv2.resize(img, (frameWidth, frameHeight))
    imgContour = img.copy()

    imgPreprocessed = preProcessing(img)
    coord_points = getContours(imgPreprocessed)

    imgWarp = getWarp(img, coord_points)

    # Rotate the video frame since webcam faces upside up down
    img_rotate_90_clockwise = cv2.rotate(imgWarp, cv2.ROTATE_180)

    cv2.imshow("Video", img_rotate_90_clockwise)

    if cv2.waitKey(10) & 0xFF == ord(' '):
        break