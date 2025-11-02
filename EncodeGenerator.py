# EncodeGenerator.py
import cv2
import face_recognition
import pickle
import os

# Folder containing images
folderImagePath = 'images'
imgList = []
studentIds = []

for filename in os.listdir(folderImagePath):
    path = os.path.join(folderImagePath, filename)
    img = cv2.imread(path)
    if img is not None:
        imgList.append(img)
        studentIds.append(os.path.splitext(filename)[0])

print(f"Total images found: {len(imgList)}")
print("Student IDs:", studentIds)

def findEncodings(imagesList):
    encodeList = []
    for idx, img in enumerate(imagesList):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
            print(f"[{idx+1}/{len(imagesList)}] Encoded: {studentIds[idx]}")
        else:
            print(f"[{idx+1}/{len(imagesList)}] ⚠️ No face found in {studentIds[idx]}")
    return encodeList

print("Encoding started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = (encodeListKnown, studentIds)
print("Encoding complete.")

# Save encodings
with open("Encodefile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)

print("✅ Encodefile.p saved successfully!")
