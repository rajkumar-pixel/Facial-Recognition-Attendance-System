import cv2
import face_recognition
import pickle
import os

# importing student images
folderPath = 'Images' 
pathList = os.listdir(folderPath)
print(pathList)  # List all files in folder
imgList = []

studentIds = []

# Read all mode images from the folder and append to the list
for path in pathList:
    
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    #print(path)
    #print(os.path.splitext(path)[0]) #splits the text and the png (removes the dot and png)
print(studentIds) #extracted the student id 

#for encoding creating a function 
def findEncodings(imagesList):
    encodeList = []
    
    for img in imagesList:      #open cv uses bgr face recog lib uses rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList


print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
#print(encodeListKnown)
print("Encoding Complete")


file = open("EncodeFile.p,",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")
