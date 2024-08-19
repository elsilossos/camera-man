import cv2
import threading
import queue 
import time 
import numpy

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# set up queues
result_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)
crop_result_queue = queue.Queue()

# define the worker thread function
def face_tracker(frame_queue, result_queue, aspectR_w, aspectR_h):
    while True:
        frame = frame_queue.get()

        # exit condition
        if frame is None: break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            result_queue.put(faces)

        # use the crop function to ... crop and put it in the outpu-queue 
        crop_x, crop_y, crop_w, crop_h = mk_crop_target(faces, aspectR_w=aspectR_w, aspectR_h=aspectR_h)
        crop_result_queue.put([crop_x, crop_y, crop_w, crop_h])

        frame_queue.task_done()


# define function that determines the ideal cropping based on facetracking
def mk_crop_target(faces, aspectR_h, aspectR_w, max_crop_ratio=0.3, bg_factor=1):

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

        # get a multiple of the face hight as a the defining factor for crop size
        crop_h = face_h * 4                         # !!! might have to make that bigger
        if crop_h < aspectR_h * max_crop_ratio:
            crop_h = aspectR_h * max_crop_ratio     # makes sure the crop is not too aggressive (adjustable)
        if crop_h > aspectR_h * 0.95: 
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
        crop_w = aspectR_w / aspectR_h * crop_h

        # get x, y coordinates for the cropping
        crop_x = eyes_x - crop_w/2          # cause one face should be centered
        crop_y = eyes_y - crop_h * (1/3)    # attemt of leverling the eyes at golden cut hight

        # now let's check whether we are in bounds and adjust if necessary
        '''if crop_x < 0: crop_x = 0
        if crop_y < 0: crop_y = 0
        if crop_x + crop_w > aspectR_w: crop_x = crop_x - abs(crop_x + crop_w - aspectR_w)
        if crop_y + crop_h > aspectR_h: crop_y = crop_y - abs(crop_y + crop_h - aspectR_h)'''

    # logic for more than 1 face
    elif len(faces) > 1: 
        
        # get those faces
        face_list = []
        for i in range(len(faces)):
            face_x, face_y, face_w, face_h = faces[i]
            face_list.append([face_x, face_y, face_w, face_h])
        
        # get biggest (longest...lol) faces in frame if more than two faces
        if len(face_list) > 2:
            face_list = sorted(face_list, key=lambda x: x[3])
            heights = [x[3] for x in face_list]
            avg_face_length = sum(heights) / len(heights)
            face_list = [face for face in face_list if face[3] >= avg_face_length * bg_factor]  # !!!

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
        eyes_ys = sorted([x[1] for x in eyes])
        eye_distance = eyes_ys[-1] - eyes_ys[0]
        crop_x = eyes_ys[0] - eye_distance
        crop_w = eye_distance * 3

        # the 'too-big-or-too-small-for-the-frame-check'
        if crop_w < aspectR_w * max_crop_ratio:
            crop_w = aspectR_w * max_crop_ratio     # makes sure the crop is not too aggressive (adjustable)
        if crop_w > aspectR_w * 0.95: 
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


# initialise secundary thread
# Get the frame width and height
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('Width: ', frame_width, '\nHeight: ', frame_height)
face_tracker_thread = threading.Thread(target=face_tracker, args=(frame_queue, result_queue, frame_width, frame_height))
face_tracker_thread.start()

# list to store faces
faces = []

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
        
    if len(faces) > 0: 
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Check if there's a result from the worker thread
    if not crop_result_queue.empty():
        rect = result_queue.get_nowait()
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)

