import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import re
import subprocess


'''def list_cameras():
    # Use ffmpeg to list video devices (works on Windows, Linux, macOS)
    cmd = 'ffmpeg -f avfoundation -list_devices true -i ""'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, _ = process.communicate()

    # Parse the output
    devices = []
    for line in stdout.decode().split('\n'):
        if '[AVFoundation input device @' in line:
            match = re.search(r'\[(.*?)\] (.*)', line)
            if match:
                devices.append(match.group(2).strip())

    return devices

print(list_cameras())
exit()'''

# Open a connection to the default camera (camera 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Set video frame width and height (optional)
'''cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)'''

# Get the width and height of the frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Open the virtual camera
with pyvirtualcam.Camera(width, height, fps=30, fmt=PixelFormat.BGR) as cam:
    print(f'Using virtual camera: {cam.device}')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break

        # Optionally process the frame (for example, convert to grayscale)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Send the frame to the virtual camera
        cam.send(frame)

        # Wait for the next frame
        cam.sleep_until_next_frame()

        # Optionally display the frame in a window (commented out here)
        cv2.imshow('Frame', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
