import numpy as np
import cv2
 
def nothing(x):
    pass
 
cap = cv2.VideoCapture(0)
 
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
while True:        
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])
 
        mask = cv2.inRange(hsv, lower_range, upper_range)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
 
        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.GaussianBlur(gray,(5, 5),cv2.BORDER_DEFAULT)
 
        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred,
                        cv2.HOUGH_GRADIENT, 1, 100, param1 = 50,
                    param2 = 30, minRadius = 30, maxRadius = 250)
 
        # Draw circles that are detected.
        if detected_circles is not None:
 
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
 
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
 
                # Draw the circumference of the circle.
                cv2.circle(frame, (a, b), r, (0, 255, 0), 5)
 
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(frame, (a, b), 1, (0, 0, 255), 5)
 
        # we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # stack the mask, orginal frame and the filtered result
        stacked = np.hstack((frame,mask_3,res))
    
        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
 
 
        if cv2.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
 
cap.release()
cv2.destroyAllWindows()
