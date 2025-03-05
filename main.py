import cv2
import os
import pickle

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
print("Background Shape:", imgBackground.shape)  # Should be (545, 972, 3)

# Define placement region
x1, y1 = 41, 112  # Top-left corner of the webcam overlay
x2, y2 = 529, 486  # Bottom-right corner
webcam_width = x2 - x1  # 488
webcam_height = y2 - y1  # 374

X1,Y1=616,40
X2,Y2=911,502

node_image_width=X2-X1 # 295
node_image_height=Y2-Y1 # 460


#importing the mode images into the List
folderModepath=(r'Resources\modes')
modePathList = os.listdir(folderModepath)
imgModeList=[]
print(modePathList)

for filename in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModepath,filename)))

print(len(imgModeList))

#load the encoding file
file = open('Encodefile.p','rb')
encodeListknownwithIds = pickle.load(file)
file.close()

encodeListknown , studentIds = encodeListknownwithIds
print(studentIds)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture webcam feed.")
        continue

    # Resize webcam to fit within the selected area
    img_resized = cv2.resize(img, (webcam_width, webcam_height))

    # Resize node image  to fit within the selected area
    node_img_resized = cv2.resize(imgModeList[1], (node_image_width, node_image_height))

    # Copy background image
    imgBackgroundCopy = imgBackground.copy()

    # Overlay resized webcam feed onto the background
    imgBackgroundCopy[y1:y2, x1:x2] = img_resized  

    imgBackgroundCopy[Y1:Y2, X1:X2] = node_img_resized

    # Display the final result
    cv2.imshow("Face Attendance", imgBackgroundCopy)

    # Exit 
    # 
    # on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
