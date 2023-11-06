import cv2
import numpy as np

# Function to detect and filter red objects in a frame
def detect_red(frame):
    global center_x, center_y
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the red color
    lower_red = np.array([0, 150, 100])
    upper_red = np.array([10, 255, 255])
    
    # Create a mask for red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours in the red color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Approximate the shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get area
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        # Draw the shape
        cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
        
        # Get the coordinates for putting text
        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + (w / 2))
        y_mid = int(y + (h / 2))
        
        # Calculate the center point of the bounding rectangle and print
        center_x = int(x + (w / 2))
        center_y = int(y + (h / 2))
        print(f"Center Point: ({center_x}, {center_y})")

        # Draw a circle at the center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 0), -1)

        print("Detected Edges:" + str(len(approx)))
        # Determine the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) >= 5 and len(approx) < 8:
            shape = "Unidentified"
        else:
            shape = "Circle"
        
        # Put the text on the frame
        cv2.putText(frame, shape, (x_mid, y_mid), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    
    return frame

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

if cap.isOpened == False:
    print("Camera is not working.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Call the function to detect and filter red objects
    result = detect_red(frame)

    # Display the frame with detected shapes
    cv2.imshow("CAM", result)
    #search()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()