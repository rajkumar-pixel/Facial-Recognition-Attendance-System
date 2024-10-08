import cv2
import os

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

#print(len(imgModeList))

while True:
    success, img = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Overlay webcam image onto the background
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808:808+414] = imgModeList[1]

    # Display the updated background with the webcam feed
    cv2.imshow("Face Attendance", imgBackground)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
