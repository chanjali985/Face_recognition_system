import cv2
import face_recognition
import pickle
import os

#importing the students images

folderImagePath='images'
folderImagelist= os.listdir(folderImagePath)
imgList=[]
studentIds=[]


for filename in folderImagelist:
    imgList.append(cv2.imread(os.path.join(folderImagePath,filename)))
    #print(filename)
   # print(os.path.splitext(filename)[0])
    studentIds.append(os.path.splitext(filename)[0])
    
    
print(len(imgList))
print(studentIds)


#loop through the each images and encode every single image we have
#Encoding steps:- 1. change the color from BGR to RGB 
def findEncodings(imagesList):
    encodeList=[]
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
        
    return encodeList
print("Encoding started ")
encodeListknown=findEncodings(imgList)
#print(encodeListknown)

# for endoings associated to the ids
encodeListknownwithIds = (encodeListknown,studentIds)

print("Encoding complete")
        
# need to save these encodings in the pickle file for later use

file  = open("Encodefile.p",'wb')
pickle.dump(encodeListknownwithIds,file)


print("file saved")
file.close()
