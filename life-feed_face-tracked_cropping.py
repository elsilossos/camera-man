import cv2
import threading
import queue 
import time 
import numpy as np

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# set up queues
result_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)
frame_queue2 = queue.Queue(maxsize=1)
#crop_result_queue = queue.Queue()
preview = True

# define the worker thread function
def face_tracker(frame_queue, result_queue, aspectR_w, aspectR_h):
    while True:
        frame = frame_queue.get()

        # exit condition
        if frame is None: break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40))
        
        if len(faces) > 0:
            result_queue.put(faces)

        # use the crop function to ... crop and put it in the outpu-queue 
        #crop_x, crop_y, crop_w, crop_h = mk_crop_target(faces, aspectR_w=aspectR_w, aspectR_h=aspectR_h)
        #crop_result_queue.put([crop_x, crop_y, crop_w, crop_h])

        frame_queue.task_done()


# define function that determines the ideal cropping based on facetracking
def mk_crop_target(faces, aspectR_h, aspectR_w, max_crop_ratio=0.3, bg_factor=1):
    '''define function that determines the ideal cropping based on facetracking. 
    Returns a crop suggestion as a tuple (x,y,w,h)'''

    # logic for one face in frame
    if len(faces) == 1:

        # get that one face
        face_x, face_y, face_w, face_h = faces[0]

        # calculate nose x, y ... not sure if i'll use it
        nose_x = int(face_x + face_w/2)
        nose_y = int(face_y + face_h/2)

        # calculate eyes-center-point
        eyes_x = int(face_x + face_w/2)
        eyes_y = int(face_y + face_h * (1/3))

        # calculate zoom_coeff new, so that the zooming isn't as jumpy as the face-detection
        global zoom_coeff_h
        zoom_coeff_h = (zoom_coeff_h * 29 + face_h) / 30

        # get a multiple of the face hight as a the defining factor for crop size
        crop_h = zoom_coeff_h * 4                         

        # check minimum zoom
        if crop_h < aspectR_h * max_crop_ratio:
            crop_h = aspectR_h * max_crop_ratio     # makes sure the crop is not too aggressive (adjustable)

        # check if zooming is necessary, cause face so close
        if crop_h > aspectR_h * 0.98: 
            crop_h = aspectR_h   # makes sure the crop is never bigger than the original
            crop_w = aspectR_w
            crop_x = 0
            crop_y = 0
            # ensure ints are send
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            crop_w = int(crop_w)
            crop_h = int(crop_h)
            return crop_x, crop_y, crop_w, crop_h       
        
        # calc width out of hight
        crop_w = aspectR_w / aspectR_h * crop_h

        # get x, y coordinates for the cropping
        crop_x = eyes_x - crop_w/2          # cause one face should be centered         !!! This is where some thought into making the whole thing also follow thirds on the x-axis needs to go...
        crop_y = eyes_y - crop_h * (1/3)    # attemt of leverling the eyes at golden cut hight

        # now let's check whether we are in bounds and adjust if necessary
        if crop_x < 0: crop_x = 0
        if crop_y < 0: crop_y = 0
        if crop_x + crop_w > aspectR_w: crop_x = crop_x - abs(crop_x + crop_w - aspectR_w)
        if crop_y + crop_h > aspectR_h: crop_y = crop_y - abs(crop_y + crop_h - aspectR_h)

    #############################################
    #############################################
    #############################################
    #############################################
    #############################################

    # logic for more than 1 face        
    elif len(faces) > 1: 
        
        # get those faces
        face_list = []
        for face in faces:
            face_x, face_y, face_w, face_h = face

            face_list.append([face_x, face_y, face_w, face_h])

        # sort for face width
        face_list.sort(key=lambda face: face[2], reverse=True)  

        # eliminate all suspiciously small faces
        face_list = [face for face in face_list if face[2] > face_list[0][2]/2]             # !!! untested

        # get average hight value out of the average eye heights
        eyes = []
        for face in face_list:
            
            # get that one face
            face_x, face_y, face_w, face_h = face

            # calculate eyes-center-point
            eyes_x = int(face_x + face_w/2)
            eyes_y = int(face_y + face_h * (1/3))
            eyes.append([eyes_x, eyes_y])  # store it for later


        # set width (x and w) so that the most left and most right are at the 3rd sweetspots
        eyes_xs = sorted([x[0] for x in eyes])
        eye_distance = eyes_xs[-1] - eyes_xs[0]
        # calculate zoom_coeff_w new, so that the zooming isn't as jumpy as the face-detection
        #global zoom_coeff_w
        #zoom_coeff_h = (zoom_coeff_h * 29 + eye_distance) / 30         # !!! Hier sollte es eigentlich beim Partnerzoom etwaw smoother zugehen, falls mal einer nicht getrackt wird, war aber eher eine Katastrophe....
        crop_w = eye_distance * 3
        crop_x = eyes_xs[0] - eye_distance

        # the 'too-big-or-too-small-for-the-frame-check'
        # too small...?
        if crop_w < aspectR_w * max_crop_ratio:
            crop_w = aspectR_w * max_crop_ratio     # makes sure the crop is not too aggressive (adjustable)

        # too big...?? 
        if crop_w > aspectR_w * 0.98: 
            crop_h = aspectR_h   # makes sure the crop is never bigger than the original
            crop_w = aspectR_w
            crop_x = 0
            crop_y = 0
            # ensure ints are send
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            crop_w = int(crop_w)
            crop_h = int(crop_h)
            return crop_x, crop_y, crop_w, crop_h              
        
        crop_h = aspectR_h / aspectR_w * crop_w
        
        # get average eye hight and extrapolate 3rd for crop_y
        avg_y = sum([x[1] for x in eyes]) / len(eyes)
        crop_y = avg_y - crop_h * (1/3)

        # make list with face y's and make sure the the lowest and highest face are within thirds, or else...
        crop_center = [crop_x + crop_h/2, crop_y + crop_h/2]
        crop_center_new = crop_center.copy()
        crop_h_new = crop_h

        # make the list
        eyes_ys = sorted([y[1] for y in eyes])
        
        # highest eyes check
        if eyes_ys[0] < crop_center[1] - crop_h_new*(1/6):
            # stretch the frame up so that the third includes the highes face:
            distance = abs(eyes_ys[0] - (crop_center[1] - crop_h_new*(1/6)))
            crop_h_new = crop_h_new + distance
            crop_center_new[1] = crop_center_new[1] + crop_center_new[1] * (distance/2)

        # lowest eyes check 
        if eyes_ys[-1] > crop_center[1] + crop_h_new*(1/6):
            # stretch the fram down so the lowest face fits
            distance = abs(eyes_ys[0] - (crop_center[1] - crop_h_new*(1/6)))
            crop_h_new = crop_h_new + distance
            crop_center_new[1] = crop_center_new[1] - crop_center_new[1] * (distance/2)

        # bring it home in case necessary
        if crop_center_new != crop_center or crop_h_new != crop_h:
            crop_h = crop_h_new
            crop_w = aspectR_w / aspectR_h * crop_h
            crop_x = crop_center_new[0] - crop_w/2
            crop_y = crop_center_new[1] - crop_h/2


        # let's check again, whether we're too big...
        # too big...?? 
        if crop_w > aspectR_w * 0.98: 
            crop_h = aspectR_h   # makes sure the crop is never bigger than the original
            crop_w = aspectR_w
            crop_x = 0
            crop_y = 0
            # ensure ints are send
            crop_x = int(crop_x)
            crop_y = int(crop_y)
            crop_w = int(crop_w)
            crop_h = int(crop_h)
            return crop_x, crop_y, crop_w, crop_h       



        # now let's check whether we are in bounds and adjust if necessary
        if crop_x < 0: crop_x = 0
        if crop_y < 0: crop_y = 0
        if crop_x + crop_w > aspectR_w: crop_x = crop_x - abs(crop_x + crop_w - aspectR_w)
        if crop_y + crop_h > aspectR_h: crop_y = crop_y - abs(crop_y + crop_h - aspectR_h)

    # logic for no faces
    else: 
        # if no faces -> zoom out!
        crop_x = 0
        crop_y = 0
        crop_w = aspectR_w
        crop_h = aspectR_h

    # ensure ints are send
    crop_x = int(crop_x)
    crop_y = int(crop_y)
    crop_w = int(crop_w)
    crop_h = int(crop_h)
    
    return crop_x, crop_y, crop_w, crop_h
            

# define logic to zoom and pan and not to zoom and pan
def crop_dscn(crr_crop: list, lv_crop: list, zoom_sens=5, pan_sens=1, tilt_sens=5):
    ''' logic to zoom and pan and not to zoom and pan
    zoom_sens is default 10 for 10%, but can be altered.
    Returns a sensible crop target (x,y,w,h)'''

    # make center points for more esoteric (?) comparisons
    crr_center = [crr_crop[0]+crr_crop[2]/2, crr_crop[1]+crr_crop[3]/2]
    lv_center =  [lv_crop[0]+lv_crop[2]/2, lv_crop[1]+lv_crop[3]/2]
    
    # first let's decide whether to zoom or not:
    # if the zoom is more than zoom_sens% (default: 5%) different from the current...
    if abs(crr_crop[2] - lv_crop[2]) > crr_crop[2] * (zoom_sens/100): 
        zoom = True
        
        # a more elaborate buffer to not make the zoom to jumpy could be put here:
        #...
        #...
        #...

        # set width and height to current detection
        tg_w, tg_h = lv_crop[2], lv_crop[3]

    # ...otherwhise to nothing
    else: 
        zoom = False
        tg_w, tg_h = crr_crop[2], crr_crop[3]

    # now lets position the frame and decide whether we want to pan
    if abs(crr_center[0] - lv_center[0]) > crr_crop[2] * (pan_sens/100):
        # off to the right
        if crr_center[0] - lv_center[0] < 0:
            tg_center_x = crr_center[0] + abs(crr_center[0] - lv_center[0]) - crr_crop[2] * (pan_sens/100)

        # off to the left
        elif crr_center[0] - lv_center[0] > 0:
            tg_center_x = crr_center[0] - abs(crr_center[0] - lv_center[0]) + crr_crop[2] * (pan_sens/100)

        # make sure to stay in frame on the RIGHT with the target
        if tg_center_x + tg_w/2 > frame_width: 
            tg_center_x = tg_center_x - abs(tg_center_x + tg_w/2 - frame_width)
        
        # make sure to stay in fram on the LEFT ...
        elif tg_center_x - tg_w/2 < 0: 
            tg_center_x = tg_center_x + abs(tg_center_x - tg_w/2)


    else: 
        tg_center_x = crr_center[0]

    
    # should we tilt?
    if abs(crr_center[1] - lv_center[1]) > frame_height * (tilt_sens/100):
        tg_center_y = lv_center[1]

        # make sure to stay in frame at the BOTTOM with the target
        if tg_center_y + tg_h/2 > frame_height: 
            tg_center_y = tg_center_y - abs(tg_center_y + tg_h/2 - frame_height)
        
        # make sure to stay in fram at the top...
        elif tg_center_y - tg_h/2 < 0: 
            tg_center_y = tg_center_y + abs(tg_center_y - tg_h/2)

    else: 
        tg_center_y = crr_center[1]


    # lastly we recolve the center points with the height and width to desired x and y
    tg_x = tg_center_x - tg_w/2
    tg_y = tg_center_y - tg_h/2

    # ensure ints are send
    tg_x = int(tg_x)
    tg_y = int(tg_y)
    tg_w = int(tg_w)
    tg_h = int(tg_h)

    return tg_x, tg_y, tg_w, tg_h


# define a function that turns the current frame and the targetframe into the next frame
def cameraman(crr_crop: list, tg_crop: list):
    '''a function that turns the current frame and the targetframe into the next frame
    practically it is zooming, panning and tilting one pixel at a time. Call in loop.
    Returns tuple with cropping info (x,y,w,h)'''

    # make center points for more esoteric (?) comparisons
    crr_center = [crr_crop[0]+crr_crop[2]/2, crr_crop[1]+crr_crop[3]/2]
    tg_center =  [tg_crop[0]+tg_crop[2]/2, tg_crop[1]+tg_crop[3]/2]
    mv_center_x = crr_center[0]
    mv_center_y = crr_center[1]

    # zoom if not on target
    if crr_crop[3] != tg_crop[3]:
        dist = abs(crr_crop[3] - tg_crop[3])
        # calc how fast we habe to tilt based on dist
        '''if dist > 50: speed = 10
        elif dist > 35: speed = 8
        elif dist > 25: speed = 6
        elif dist > 10: speed = 4
        else: speed = 2'''
        speed = 2
        
        # find hight
        if crr_crop[3] > tg_crop[3]: mv_h = crr_crop[3] - speed
        else: mv_h = crr_crop[3] + speed

        # calculate width in ratio to original input frame
        mv_w = int(frame_width / frame_height * mv_h)
    else: 
        mv_w = crr_crop[2]
        mv_h = crr_crop[3]

    # pan if not on target
    if crr_center[0] != tg_center[0]:
        dist = abs(crr_center[0] - tg_center[0])
        # calc how fast we habe to pan based on dist
        if dist > 75: speed = 15
        elif dist > 50: speed = 10
        elif dist > 35: speed = 7
        elif dist > 25: speed = 4
        elif dist > 10: speed = 2
        else: speed = 1
        if crr_center[0] > tg_center[0]: mv_center_x = crr_center[0] - speed
        elif crr_center[0] < tg_center[0]: mv_center_x = crr_center[0] + speed
        else: mv_center_x = crr_center[0]

        # make sure to stay in frame on the RIGHT with the target
        if mv_center_x + mv_w/2 > frame_width: 
            mv_center_x = mv_center_x - abs(mv_center_x + mv_w/2 - frame_width)
        
        # make sure to stay in fram on the LEFT ...
        elif mv_center_x - mv_w/2 < 0: 
            mv_center_x = mv_center_x + abs(mv_center_x - mv_w/2)

    # tilt if not on target
    if crr_center[1] != tg_center[1]:
        dist = abs(crr_center[1] - tg_center[1])
        # calc how fast we habe to tilt based on dist
        if dist > 50: speed = 15
        elif dist > 30: speed = 10
        elif dist > 20: speed = 7
        elif dist > 10: speed = 4
        elif dist > 5: speed = 2
        else: speed = 1
        if crr_center[1] > tg_center[1]: mv_center_y = crr_center[1] - speed
        elif crr_center[1] < tg_center[1]: mv_center_y = crr_center[1] + speed
        else: mv_center_y = crr_center[1]

        # make sure to stay in frame at the BOTTOM with the target
        if mv_center_y + mv_h/2 > frame_height: 
            mv_center_y = mv_center_y - abs(mv_center_y + mv_h/2 - frame_height)
        
        # make sure to stay in fram at the top...
        elif mv_center_y - mv_h/2 < 0: 
            mv_center_y = mv_center_y + abs(mv_center_y - mv_h/2)


    # lastly we recolve the center points with the height and width to desired x and y
    mv_x = mv_center_x - mv_w/2
    mv_y = mv_center_y - mv_h/2

    # ensure ints are send
    mv_x = int(mv_x)
    mv_y = int(mv_y)
    mv_w = int(mv_w)
    mv_h = int(mv_h)

    return mv_x, mv_y, mv_w, mv_h


# define a fuction that will finally zoom in
def zoom(frame, x, y, width, height):
    # Extract the region of interest (ROI)
    roi = frame[y:y+height, x:x+width]
    
    # Get the original frame dimensions
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate the aspect ratio of the ROI
    roi_aspect_ratio = width / float(height)
    
    # Calculate the aspect ratio of the original frame
    frame_aspect_ratio = frame_w / float(frame_h)
    
    # Determine the new width and height to maintain aspect ratio
    if roi_aspect_ratio > frame_aspect_ratio:
        # ROI is wider than the frame, fit width
        new_width = frame_w
        new_height = int(frame_width / roi_aspect_ratio)
    else:
        # ROI is taller than the frame, fit height
        new_height = frame_h
        new_width = int(frame_h * roi_aspect_ratio)
    
    # Resize the ROI to the new dimensions
    zoomed_roi = cv2.resize(roi, (new_width, new_height))
    
    # Create a black canvas with the same size as the original frame
    canvas = cv2.resize(zoomed_roi, (frame_w, frame_h))
    
    return canvas





# Open a connection to the default camera (camera 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

trackers = []

'''# Set video frame width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)'''


# Get the frame width and height
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
zoom_coeff_h = frame_height / 4
#zoom_coeff_w = frame_width
print('Width: ', frame_width, '\nHeight: ', frame_height)

# set initial crops to max frame, both current (crr_crop) and target (tg_crop)
crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h = 0, 0, frame_width, frame_height
tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h = 0, 0, frame_width, frame_height
lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h = 0, 0, frame_width, frame_height

# initialise secundary thread
face_tracker_thread = threading.Thread(target=face_tracker, args=(frame_queue, result_queue, frame_width, frame_height))
face_tracker_thread.start()

# list to store faces and timer to reset it
faces = []
empty_since = time.time() 

# start time and check intervall
last_check = time.time()
min_check_intervall = 0.5


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break

    # Check if the worker thread is ready for a new frame 
    # and our minimum checking intervall has passed
    if frame_queue.empty() and time.time() - last_check > min_check_intervall:
        frame_queue.put(frame)
        last_check = time.time()
    
    # Check if there's a result from the worker thread
    if not result_queue.empty():
        faces = result_queue.get_nowait()
        empty_since = time.time()

    # reset faces to empty if there were no face detections for 3 seconds. 
    if time.time() - empty_since > 5: faces = []

    preview_frame = frame.copy()

    # calculate the frame if there is faces in the frame  
    if len(faces) > 0: 
        for face in faces:
            x, y, w, h = face
            if preview: cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # blue square for faces

        # calculate the cropping based on the current faces results
        lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h = mk_crop_target(faces, aspectR_w=frame_width, aspectR_h=frame_height)
        if preview: cv2.rectangle(preview_frame, (lv_crop_x, lv_crop_y), (lv_crop_x + lv_crop_w, lv_crop_y + lv_crop_h), (0, 255, 0), 2)        # green rectangle for current ideal

    # reset the frame to max view if there are no faces
    else: 
        #crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h = 0, 0, frame_width, frame_height
        tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h = 0, 0, frame_width, frame_height
        lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h = 0, 0, frame_width, frame_height
        

    # calculate the target (tg_crop) with crop_desc(). Should often stay static.
    tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h = crop_dscn(
                crr_crop=[tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h], 
                lv_crop= [lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h])
    if preview: cv2.rectangle(preview_frame, (tg_crop_x, tg_crop_y), (tg_crop_x + tg_crop_w, tg_crop_y + tg_crop_h), (255, 168, 0), 2)        # orange rectangle for target ideal
    
    # paint based on crop_target by passing crop_desc() into cameraman(), updating crr_crop-values
    crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h = cameraman(
        crr_crop=[crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h],
        tg_crop=[tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h])
    
    # paint frame for testing
    if preview: cv2.rectangle(preview_frame, (crr_crop_x, crr_crop_y), (crr_crop_x + crr_crop_w, crr_crop_y + crr_crop_h), (0, 0, 255), 2)  # blue square

    try: zoomed_frame = zoom(frame, crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h)
    except: 
        print('Woups!')
        continue

    # Display the resulting frame
    if preview: cv2.imshow('Preview', preview_frame)        # this needs to be tied to a condition and need live kill switch

    # Display the zoomed in
    cv2.imshow('Zoom', zoomed_frame)            # this needs to be a systemic out (as alternative)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)

