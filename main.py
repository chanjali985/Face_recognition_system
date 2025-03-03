import cv2
import os

# 1) Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set webcam width
cap.set(4, 480)  # Set webcam height

# 2) Load your background image
background_path = r'C:\Users\anjali\Downloads\Face_recognition_system\Resources\background.PNG'
imgBackground = cv2.imread(background_path)
if imgBackground is None:
    print("Error: Could not load background image. Check the file path!")
    exit()
print("Background Shape:", imgBackground.shape)  # e.g., (545, 972, 3)

# 3) Import mode images into a list
folderModePath = r'Resources\modes'
Modepathlist = os.listdir(folderModePath)
imgModeList = []
print("Mode files:", Modepathlist)
for path in Modepathlist:
    full_path = os.path.join(folderModePath, path)
    imgModeList.append(cv2.imread(full_path))
print("Number of mode images loaded:", len(imgModeList))

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture webcam feed.")
        continue

    # Create a fresh copy of the background for each frame
    imgBackgroundCopy = imgBackground.copy()

    # Overlay the webcam feed onto the background copy.
    # Choose a region that fits within your background.
    # For example, starting at row 50 and column 55 gives:
    # Region: rows 50 to 50+480, columns 55 to 55+640
    imgBackgroundCopy[56:56+300, 63:63+220] = img

    # Display the final result
    cv2.imshow("Face Attendance", imgBackgroundCopy)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
