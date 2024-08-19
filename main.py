import cv2
import threading
import queue 
import time 
import numpy as np
import pyvirtualcam 
import settings as stt

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# set up queues
result_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)
frame_queue2 = queue.Queue(maxsize=1)
status_queue = queue.Queue(maxsize=1)
settings = stt.get_settings()
tech_preview = False                          # !!! settings
preview = True







#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################









# define the worker thread function
def face_tracker_thread(frame_queue, result_queue, aspectR_w, aspectR_h):
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




def are_frames_equal(frame1, frame2):
    # Check if the shapes are the same
    '''if frame1.shape != frame2.shape:
        return False'''
    
    # Check if all pixel values are the same
    return np.array_equal(frame1, frame2)




def is_frame_uniform(frame):
    # Compare the entire frame with its first pixel value
    return np.all(frame == frame[0, 0])





def change_tracker_thread(frame_queue2, status_queue):
    old_frame = frame_queue2.get()

    if old_frame is None:
        frame_queue2.put(None)
    
    while True:
        new_frame = frame_queue2.get()

        # exit condition
        if new_frame is None: break

        # check old and new frame
        equal = are_frames_equal(old_frame, new_frame)
        mono = is_frame_uniform(new_frame)
        
        # if there is a non mono changed frame, the status will be set to ten, which will then slowly degrade in the main function
        if not equal and not mono: 
            status_queue.put(10)


        old_frame = new_frame

        frame_queue2.task_done()




def add_padding(frame, target_width, target_height):
    """Add black padding to the frame to reach the target width and height."""
    h, w, _ = frame.shape

    # Calculate the padding needed
    delta_w = target_width - w
    delta_h = target_height - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Add black padding (0 for black color)
    padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_frame




def resize_with_aspect_ratio(frame, target_width, target_height):
    """Resize the frame while maintaining the aspect ratio, then pad to match the target dimensions."""
    h, w = frame.shape[:2]

    # Calculate the scaling factor
    scale = min(target_width / w, target_height / h)
    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize the frame with the new dimensions
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Add padding to reach the target size without altering the aspect ratio
    padded_frame = add_padding(resized_frame, target_width, target_height)

    return padded_frame




def big_split(frame1, frame2, margin=10, pc_size=0.2):
    # Resize frame2 to fit within the resolution of frame1 without distorting the aspect ratio
    frame2_resized = resize_with_aspect_ratio(frame2, frame1.shape[1], frame1.shape[0])

    # Resize frame1 to be a smaller version for the bottom-left corner
    small_height = int(frame1.shape[0] * pc_size)
    aspect_ratio = frame1.shape[1] / frame1.shape[0]
    small_width = int(small_height * aspect_ratio)

    small_frame1 = cv2.resize(frame1, (small_width, small_height))

    # Create a copy of the resized frame2 as the background
    combined_frame = frame2_resized.copy()

    # Calculate position for small_frame1 in the bottom-left with a margin
    x_offset = margin
    y_offset = frame1.shape[0] - small_height - margin

    # Place small_frame1 on top of combined_frame at the calculated position
    combined_frame[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_frame1

    return combined_frame





def small_split(frame1, frame2, margin=10, pc_size = 0.3):
    # Resize frame2 to be a slightly larger version for the bottom-right corner
    small_height = int(frame1.shape[0] * pc_size)  # 40% of frame1's height
    aspect_ratio = frame2.shape[1] / frame2.shape[0]  # width / height of frame2
    small_width = int(small_height * aspect_ratio)
    
    small_frame2 = cv2.resize(frame2, (small_width, small_height))
    
    # Create a copy of frame1 (the background)
    combined_frame = frame1.copy()

    # Calculate position for small_frame2 in the bottom-right with a margin
    x_offset = frame1.shape[1] - small_width - margin
    y_offset = frame1.shape[0] - small_height - margin
    
    # Place small_frame2 on top of combined_frame at the calculated position
    combined_frame[y_offset:y_offset + small_height, x_offset:x_offset + small_width] = small_frame2
    
    return combined_frame












#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################
#################################








# Open a connection to the default camera (camera 0)
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()
if not cap2.isOpened():
    print("Error: Could not open video device")
    exit()

trackers = []


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

# initialise secundary and tertiary thread
# 1
face_tracker_thread = threading.Thread(target=face_tracker_thread, args=(frame_queue, result_queue, frame_width, frame_height))
face_tracker_thread.start()
# 2
change_tracker_thread = threading.Thread(target=change_tracker_thread, args=(frame_queue2, status_queue))
change_tracker_thread.start()

# list to store faces and timer to reset it
faces = []
empty_since = time.time() 

# start time on check intervalls
# face-tracking
last_face_check = time.time()
min_face_check_intervall = 0.5
# change-checking
last_change_check = time.time()
min_check_intervall = 1
# status-checking
status_check = time.time()
status_intervall = 1
status = 0

# Initialize pyvirtualcam
# with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=20, fourcc=544694642) as cam:          # not ready yet.... :(
while True:
    # update settings
    settings = stt.get_settings()

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break

    # Check if the worker thread is ready for a new frame 
    # and our minimum checking intervall has passed
    if frame_queue.empty() and time.time() - last_face_check > min_face_check_intervall:
        frame_queue.put(frame)
        last_face_check = time.time()
    
    # Check if there's a result from the worker thread
    if not result_queue.empty():
        faces = result_queue.get_nowait()
        empty_since = time.time()

    # reset faces to empty if there were no face detections for 3 seconds. 
    if time.time() - empty_since > 5: faces = []

    

    # calculate the frame if there is faces in the frame  
    if len(faces) > 0: 
        # calculate the cropping based on the current faces results
        lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h = mk_crop_target(faces, aspectR_w=frame_width, aspectR_h=frame_height)
        

    # reset the frame to max view if there are no faces
    else: 
        #crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h = 0, 0, frame_width, frame_height
        tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h = 0, 0, frame_width, frame_height
        lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h = 0, 0, frame_width, frame_height
        

    # calculate the target (tg_crop) with crop_desc(). Should often stay static.
    tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h = crop_dscn(
                crr_crop=[tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h], 
                lv_crop= [lv_crop_x, lv_crop_y, lv_crop_w, lv_crop_h])
    
    
    # paint based on crop_target by passing crop_desc() into cameraman(), updating crr_crop-values
    crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h = cameraman(
        crr_crop=[crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h],
        tg_crop=[tg_crop_x, tg_crop_y, tg_crop_w, tg_crop_h])
    
    # finally do the zoom
    try: frame = zoom(frame, crr_crop_x, crr_crop_y, crr_crop_w, crr_crop_h)
    except: 
        print('Woups!')
        continue
    

    #################################
    ### CHANGE MONITORING BELOW   ###
    ################################



    # degrade status
    if time.time() - status_check > status_intervall and status > 0:
        status -= 1
        print(status)
        status_check = time.time()

    # pull status from queue
    if not status_queue.empty():
        status = status_queue.get()
        status_queue.task_done()

    # get second feed
    if status > 0:
        ret2, frame2 = cap2.read()
        if not ret2:
                print("Error: Could not read frame2")
                continue

    # Check if the worker thread is ready for a new frame 
    # and our minimum checking intervall has passed
    if frame_queue2.empty() and time.time() - last_change_check > min_check_intervall:
        if status == 0: 
            # Capture frame-by-frame
            ret2, frame2 = cap2.read()
            
            if not ret2:
                print("Error: Could not read frame2")
                continue
        

        frame_queue2.put(frame2)
        last_change_check = time.time()


    # overlap frames, if necessary
    if status >= 5:
        frame = big_split(frame, frame2)
    elif status > 0:
        frame = small_split(frame, frame2)
    else: frame = frame


    # Push the processed frame to the virtual webcam
    #cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #cam.sleep_until_next_frame()

    # Display the window preview of finished frame
    if settings['preview']: cv2.imshow('Preview', frame)           



    # Display the resulting frame
    if settings['tech-preview']: 
        preview_frame = frame.copy()
        for face in faces:                          # this needs live kill switch
            x, y, w, h = face
            cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)    # blue square for faces
        cv2.rectangle(preview_frame, (lv_crop_x, lv_crop_y), (lv_crop_x + lv_crop_w, lv_crop_y + lv_crop_h), (0, 255, 0), 2)            # green rectangle for current ideal
        cv2.rectangle(preview_frame, (tg_crop_x, tg_crop_y), (tg_crop_x + tg_crop_w, tg_crop_y + tg_crop_h), (255, 168, 0), 2)          # cyan rectangle for target ideal
        cv2.rectangle(preview_frame, (crr_crop_x, crr_crop_y), (crr_crop_x + crr_crop_w, crr_crop_y + crr_crop_h), (0, 0, 255), 2)      # blue square
        cv2.imshow('Preview', preview_frame)        

    

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cap2.release()
cv2.destroyAllWindows()
frame_queue.put(None)
frame_queue2.put(None)

