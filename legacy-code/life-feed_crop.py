import cv2

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)

# Get video properties (if necessary)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the crop region (x, y, width, height)
crop_x = 100
crop_y = 50
crop_width = 400
crop_height = 300

# Create a window to display the video
cv2.namedWindow('Cropped Video', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Crop the frame
    cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    # Display the cropped frame
    cv2.imshow('Cropped Video', cropped_frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print("Webcam feed ended.")
