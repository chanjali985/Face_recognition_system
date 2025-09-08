import cv2
import os
import pickle
import face_recognition
import numpy as np

# 1) Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Original Webcam width
cap.set(4, 480)  # Original Webcam height

# 2) Load background image
background_path = r'C:\Users\anjali\Downloads\Face_recognition_system\Resources\background.PNG'
imgBackground = cv2.imread(background_path)
if imgBackground is None:
    print("Error: Could not load background image. Check the file path!")
    exit()

# Define placement region for webcam and node image
x1, y1, x2, y2 = 41, 112, 529, 486  # Webcam overlay region
X1, Y1, X2, Y2 = 616, 40, 911, 502  # Node overlay region

webcam_width, webcam_height = x2 - x1, y2 - y1
node_image_width, node_image_height = X2 - X1, Y2 - Y1

# Import mode images into a list
folderModepath = r'Resources\modes'
modePathList = os.listdir(folderModepath)
imgModeList = [cv2.imread(os.path.join(folderModepath, filename)) for filename in modePathList]

# Load the encoding file
print("Loading encoded file...")
file = open('Encodefile.p', 'rb')
encodeListknownwithIds = pickle.load(file)
file.close()

encodeListknown, studentIds = encodeListknownwithIds
print("Encode file loaded:", studentIds)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture webcam feed.")
        continue

    # **✅ FIXED: Correct image processing order**
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)   
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  

    # Detect faces & encode them
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Resize webcam and node images
    img_resized = cv2.resize(img, (webcam_width, webcam_height))
    node_img_resized = cv2.resize(imgModeList[1], (node_image_width, node_image_height))

    # Copy background image
    imgBackgroundCopy = imgBackground.copy()

    # Overlay resized webcam feed onto the background
    imgBackgroundCopy[y1:y2, x1:x2] = img_resized  
    imgBackgroundCopy[Y1:Y2, X1:X2] = node_img_resized

    # Loop through detected faces
    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurFrame):
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw rectangle for all detected faces (Red for unknown initially)
        cv2.rectangle(imgBackgroundCopy, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #  Match face inside the loop
        matches = face_recognition.compare_faces(encodeListknown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown, encodeFace)

        if matches and len(matches) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:  # Face is known
                print("Known face detected")
                cv2.rectangle(imgBackgroundCopy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for known faces
                cv2.putText(imgBackgroundCopy, f"ID: {studentIds[matchIndex]}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the final result
    cv2.imshow("Face Attendance", imgBackgroundCopy)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
