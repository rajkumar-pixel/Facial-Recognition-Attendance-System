import cv2
import os
import pickle
import face_recognition


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load background image
imgBackground = cv2.imread('Resources/background.png')

# Load images from the 'Modes' folder
folderModePath = 'Resources/Modes' 
modePathList = os.listdir(folderModePath)  # List all files in folder
imgModeList = []

# Read all mode images from the folder and append to the list
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading Encoded File ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    if not success:
        print("Failed to grab frame")
        break

    # Overlay webcam image onto the background
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]


    for encodeFace , faceLoc in zip (encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print("maches",matches)
        print("faceDis",faceDis)

    # Display the updated background with the webcam feed
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

    # Check if 'q' is pressed or the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the window is closed (handle the X button)
    if cv2.getWindowProperty("Face Attendance", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
