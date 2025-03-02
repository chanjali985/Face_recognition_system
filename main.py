import cv2

# 1) Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Webcam width
cap.set(4, 480)  # Webcam height

# 2) Load your background image
background_path = r'C:\Users\LENOVO\Downloads\Real_time_face_recognition_System\Resources\background.webp'
imgBackground = cv2.imread(background_path)

if imgBackground is None:
    print("Error: Could not load background image. Check the file path!")
    exit()

print("Background Shape:", imgBackground.shape)  # e.g., (1024, 1024, 3)

# 3) Adjusted position and size for the webcam feed
# Try these values first, then fine-tune if needed
x_start, y_start = 268, 170     # Shift the feed left/right (x), up/down (y)
frame_width, frame_height = 382, 395  # Width & height of the webcam in the placeholder

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture webcam feed.")
        continue

    # 4) Resize the webcam frame to fit your placeholder
    img_resized = cv2.resize(img, (frame_width, frame_height))

    # 5) Create a copy of the background for each loop iteration
    imgBackgroundCopy = imgBackground.copy()

    # 6) Overlay the webcam feed on the background copy
    imgBackgroundCopy[y_start : y_start + frame_height, x_start : x_start + frame_width] = img_resized

    # 7) Show the result
    cv2.imshow("Face Attendance", imgBackgroundCopy)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
