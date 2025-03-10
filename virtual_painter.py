import cv2 
import mediapipe as mp 
import os
import numpy as np

# initialize webcam video capture
cap = cv2.VideoCapture(0) # open the default webcam:index 0
# here 3,4 are property ids, 3:width 4:height
cap.set(3, 640)  # Set width to 640px
cap.set(4, 1000)  # Set height to 1000px
cap.set(10, 150)  # Set brightness

# initialize mediapipe "Hands" object for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils



#  folder for color images 
folder = 'colors'
# mylist = os.listdir(folder)
mylist = sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0]))   # sorted list of folders
# print(mylist)
 
overlist = []     
# load color images append to list
for i in mylist:
    image = cv2.imread(f'{folder}/{i}')
    print(image.shape)
    overlist.append(image)

col = [0, 0, 255]  # default color (red)
# set the initial header image :first image in list
header = overlist[0]

print(mylist)
xp, yp = 0, 0

# creating a blank canvas to draw on
canvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # capture a frame from the webcam and flip it horizontally (to create mirror image)
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # ------------------------ convert frame for hand tracking -----------------
    
    # OpenCV loads images in BGR format but MediaPipe needs RGB so convert frame to RGB 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)     #  detect hand landmarks in frame



    # -------------------------------- PROCESS DETECTED HANDS ------------------------------------
    
    lanmark = []    # store hand landmarks

    if results.multi_hand_landmarks:    #list of detected hands in the frame
        for hn in results.multi_hand_landmarks:
            # hn.landmark gives 21 key points (x, y, z) for each hand.
            # -----> for each hand, extract the 21 landmarks. 
            # -----> convert them to pixel values and store them.
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lanmark.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)
    
    # ------------------------------- DETECT HAND GESTURE -----------------
    if len(lanmark) != 0:
        
        # Check if the hand is in "selection mode" or "drawing mode"
        
        # positions of tip : index finger (8) and middle finger (12) 
        x1, y1 = lanmark[8][1], lanmark[8][2]
        x2, y2 = lanmark[12][1], lanmark[12][2]

        # condition to detect if both the index finger and middle finger are extended upwards.
        if lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
            xp, yp = 0, 0  
            print('Selection mode')

            # detect the color chosen by the hand position
            if y1 < 100:
                if 71 < x1 < 142:     # black -eraser
                    header = overlist[7]
                    col = (0, 0, 0)
                if 142 < x1 < 213:     # purple
                    header = overlist[6]
                    col = (226, 43, 138)
                elif 213 < x1 < 284:  # White
                    header = overlist[5]
                    col = (255, 255, 255)
                elif 284 < x1 < 355:  # green
                    header = overlist[4]
                    col = (0, 255, 0)
                elif 355 < x1 < 426:  # Yellow
                    header = overlist[3]
                    col = (0, 191, 255)
                elif 426 < x1 < 497:  # blue
                    header = overlist[2]
                    col = (255, 255, 0)                    
                elif 497 < x1 < 568:  # red
                    header = overlist[1]
                    col = ( 0, 0,255)
                
            # Draw a rectangle representing the selected color
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        # elif lanmark[8][2] < lanmark[6][2]:
        #     if xp == 0 and yp == 0:
        #         xp, yp = x1, y1
        #     # draw lines on the canvas when in "drawing mode"
        #     if col == (0, 0, 0):
        #         cv2.line(frame, (xp, yp), (x1, y1), col, 100, cv2.FILLED)
        #         cv2.line(canvas, (xp, yp), (x1, y1), col, 100, cv2.FILLED)
        #     cv2.line(frame, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
        #     cv2.line(canvas, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
        #     print('Drawing mode')
        #     xp, yp = x1, y1
        
        # condition to detect :index finger is up, but the middle finger is down
        elif lanmark[8][2] < lanmark[6][2] and lanmark[12][2] > lanmark[10][2]:  
            
            brush_size = 10  # default brush size
            if len(lanmark) >= 8:  # ensure landmarks exist
                x1, y1 = lanmark[8][1], lanmark[8][2]  # index finger tip
                x2, y2 = lanmark[4][1], lanmark[4][2]  # thumb tip

                # calculate distance between thumb and index finger
                thumb_index_gap = int(np.hypot(x2 - x1, y2 - y1))

                # set brush size dynamically (min: 5px, max: 50px)
                brush_size = max(5, min(thumb_index_gap // 2, 50))  

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # drawing with dynamic brush size
            if col == (0, 0, 0):  # eraser mode
                cv2.line(frame, (xp, yp), (x1, y1), col, brush_size * 3)
                cv2.line(canvas, (xp, yp), (x1, y1), col, brush_size * 3)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), col, brush_size)
                cv2.line(canvas, (xp, yp), (x1, y1), col, brush_size)
            
            print('Drawing mode')
            xp, yp = x1, y1  # update previous point

            # show brush size on screen
            cv2.putText(frame, f'Brush Size: {brush_size}', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
  
        # ----------------------------------------

    # prepare the canvas for blending with the frame
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    # use bitwise operations to blend the frame with the canvas
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    # add the header (color selection) at the top of the frame
    header = cv2.resize(header, (640, 100))
    frame[0:100, 0:640] = header

    # show the webcam frame and the canvas
    cv2.imshow('cam', frame)
    # cv2.imshow('canvas', canvas)

    # break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
