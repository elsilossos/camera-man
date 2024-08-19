import cv2
import threading
import queue 
import time 
import numpy as np


frame_queue2 = queue.Queue(maxsize=1)
status_queue = queue.Queue(maxsize=1)

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()
# Check if the cameras are opened successfully
if not cap2.isOpened():
    print("Error: Could not open video device")
    exit()


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





# FUNCTIONS

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






# initialise secundary thread
change_tracker_thread = threading.Thread(target=change_tracker_thread, args=(frame_queue2, status_queue))
change_tracker_thread.start()
last_change_check = time.time()
min_check_intervall = 1
status_check = time.time()
status_intervall = 1
status = 0



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break

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

    if status >= 5:
        picture = big_split(frame, frame2)
    elif status > 0:
        picture = small_split(frame, frame2)
    else: picture = frame

    cv2.imshow('Preview', picture)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
frame_queue2.put(None)

    
