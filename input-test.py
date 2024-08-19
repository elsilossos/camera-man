import cv2

inputs = []
i = 0

while True:
    # Open a connection to the camera (starting from index 0)
    cap = cv2.VideoCapture(i)
    
    if not cap.isOpened():
        print(f"No more video devices found after {i}.")
        break

    print(f"Found video device {i}.")
    inputs.append(cap)
    i += 1

print(f"Total video devices found: {len(inputs)}")

# Set video frame width and height (optional)
'''for cap in inputs:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)'''

while True:
    for i, cap in enumerate(inputs):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read frame from device {i}")
            break

        # Display the resulting frame
        cv2.imshow(f'Input-{i}', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all captures
for cap in inputs:
    cap.release()

cv2.destroyAllWindows()
