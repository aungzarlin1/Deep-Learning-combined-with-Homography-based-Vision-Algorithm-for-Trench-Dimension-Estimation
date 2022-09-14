import numpy as np
import cv2 
from detect import Detect

def homography(image, detected_points, method='auto'):

    if method == 'auto':
        print('method auto')
        CHECKERBOARD = (6, 8)
        criteria =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None
        img = cv2.imread(image)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                imgpoints.append(corners2)

                point4 = (int(corners2[0,0,0]), int(corners2[0,0,1]))
                point1 = (int(corners2[5,0,0]), int(corners2[5,0,1]))
                point2 = (int(corners2[-1,0,0]), int(corners2[-1,0,1]))
                point3 = (int(corners2[42,0,0]), int(corners2[42,0,1]))
            

                img = cv2.line(img, point1, point2, (0, 255, 0), 2)
                img = cv2.line(img, point2, point3, (0, 255, 0), 2)
                img = cv2.line(img, point3, point4, (0, 255, 0), 2)
                img = cv2.line(img, point1, point4, (0, 255, 0), 2)

                pts = np.array([point1, point2, point3, point4])
      

                src = np.array(pts).astype(np.float32)
                print(src.shape)
                print(src)
                reals = np.array([(0, 0),
                      (14.8, 0),
                      (14.8, 10),
                      (0, 10)], np.float32)


    
    elif method == 'manual':
        print('method manual')

        h_matrix = manual(image)
        return h_matrix

    
    elif method == 'yolo':
        print('method yolo')
        src = detected_points['sign'].reshape(4,2)
        print(src)
        reals = np.array([(0, 0), (21,0), (21, 29), (0, 29)], np.float32)
    else:
        raise ValueError('Method must be one of the followings: "auto", "manual" or "yolo".')
    

    h_matrix,_ = cv2.findHomography(reals, src);
    
    return h_matrix


matResult = None
matFinal = None
matPauseScreen = None

point = (-1, -1)
pts = []
var = 0 
drag = 0

def mouseHandler(event, x, y, flags, param):
    global point, pts, var, drag, matFinal, matResult

    if (var >= 4):                           # if homography points are more than 4 points, do nothing
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        drag = 1                             # Set it that the mouse is in pressing down mode
        matResult = matFinal.copy()          # copy final image to draw image
        point = (x, y)                       # memorize current mouse position to point var
        if (var >= 1):                       # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)             # draw a current green point
        cv2.imshow("Source", matResult)      # show the current drawing
    if (event == cv2.EVENT_LBUTTONUP and drag):  # When Press mouse left up
        drag = 0                             # no more mouse drag
        pts.append(point)                    # add the current point to pts
        var += 1                             # increase point number
        matFinal = matResult.copy()          # copy the current drawing image to final image
        if (var >= 4):                                                      # if the homograpy points are done
            cv2.line(matFinal, pts[0], pts[3], (0, 255, 0, 255), 2)   # draw the last line
            cv2.fillConvexPoly(matFinal, np.array(pts, 'int32'), (0, 120, 0, 20))        # draw polygon from points
        cv2.imshow("Source", matFinal);
    if (drag):                                    # if the mouse is dragging
        matResult = matFinal.copy()               # copy final images to draw image
        point = (x, y)                   # memorize current mouse position to point var
        if (var >= 1):                            # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)         # draw a current green point
        cv2.imshow("Source", matResult)           # show the current drawing

def mouseHandler(event, x, y, flags, param):
    global point, pts, var, drag, matFinal, matResult

    if (var >= 4):                           # if homography points are more than 4 points, do nothing
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        drag = 1                             # Set it that the mouse is in pressing down mode
        matResult = matFinal.copy()          # copy final image to draw image
        point = (x, y)                       # memorize current mouse position to point var
        if (var >= 1):                       # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)             # draw a current green point
        cv2.imshow("Source", matResult)      # show the current drawing
    if (event == cv2.EVENT_LBUTTONUP and drag):  # When Press mouse left up
        drag = 0                             # no more mouse drag
        pts.append(point)                    # add the current point to pts
        var += 1                             # increase point number
        matFinal = matResult.copy()          # copy the current drawing image to final image
        if (var >= 4):                                                      # if the homograpy points are done
            cv2.line(matFinal, pts[0], pts[3], (0, 255, 0, 255), 2)   # draw the last line
            cv2.fillConvexPoly(matFinal, np.array(pts, 'int32'), (0, 120, 0, 20))        # draw polygon from points
        cv2.imshow("Source", matFinal);
    if (drag):                                    # if the mouse is dragging
        matResult = matFinal.copy()               # copy final images to draw image
        point = (x, y)                   # memorize current mouse position to point var
        if (var >= 1):                            # if the point has been added more than 1 points, draw a line
            cv2.line(matResult, pts[var - 1], point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(matResult, point, 2, (0, 255, 0), -1, 8, 0)         # draw a current green point
        cv2.imshow("Source", matResult)           # show the current drawing

def manual(VIDEO_FILE):
    global matFinal, matResult, matPauseScreen
    key = -1;

    videoCapture = cv2.imread(VIDEO_FILE)

    height, width,_ = videoCapture.shape

    
    ratio = 640.0 / width
    dim = (int(width * ratio), int(height * ratio))
    matFrameDisplay = cv2.resize(videoCapture, dim)

    cv2.imshow(VIDEO_FILE, videoCapture)
    key = cv2.waitKey(30)

    if videoCapture is not None:
        cv2.destroyWindow(VIDEO_FILE)
        matPauseScreen = videoCapture
        matFinal = matPauseScreen.copy()
        cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Source", mouseHandler)
        cv2.imshow("Source", matPauseScreen)
        cv2.waitKey(0)
        cv2.destroyWindow("Source")

        if (len(pts) < 4):
            return

        src = np.array(pts).astype(np.float32)
        reals = np.array([(0, 0),
                            (29.5, 0),
                            (29.5, 21),
                            (0, 21)], np.float32)

        homography_matrix = cv2.getPerspectiveTransform(reals, src);
    

        cv2.waitKey(0)
        return homography_matrix