import cv2

# Load background image
background_path = r'C:\Users\anjali\Downloads\Face_recognition_system\Resources\background.PNG'
imgBackground = cv2.imread(background_path)

if imgBackground is None:
    print("Error: Could not load background image. Check the file path!")
    exit()

# Function to capture mouse click coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Clicked at X: {x}, Y: {y}")

# Display the image and capture clicks
cv2.imshow("Select Webcam Position", imgBackground)
cv2.setMouseCallback("Select Webcam Position", get_coordinates)

cv2.waitKey(0)
cv2.destroyAllWindows()
