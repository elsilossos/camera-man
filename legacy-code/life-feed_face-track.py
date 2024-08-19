import cv2
import threading
import queue 
import time 

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# set up queues
result_queue = queue.Queue()
frame_queue = queue.Queue(maxsize=1)

# define the worker thread function
def face_tracker(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()

        # exit condition
        if frame is None: break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            result_queue.put(faces)

        frame_queue.task_done()




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
face_tracker_thread = threading.Thread(target=face_tracker, args=(frame_queue, result_queue))
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
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)

