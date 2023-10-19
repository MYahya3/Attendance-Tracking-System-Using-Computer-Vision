import glob
import torch
import cv2
import numpy as np
import datetime
import os


# Yolov5s Model for Person Detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]
model.conf = 0.60

# Create a Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Replace with the correct path

# Initialize VidepCapture and get video FPS
cap = cv2.VideoCapture("cam1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Define Region of Interest Coordinates
roi_coords = [(428, 712), (578, 567), (1009, 622), (1012, 712)]
# roi_coords2 = [(529, 501), (930, 524), (914, 703), (368, 645)]

# Counter number of detections inside ROI
detections_counter = 0
# Delay Counter to ignore idle persons walk from ROI
delayCounter = 0
# Tracker if no more detection inside ROI, it reset delayCounter
resetCounter = []

# To write output video in mp4 format
# out_vid = os.path.join("images" + f"/output.mp4")
# out = cv2.VideoWriter(out_vid, cv2.VideoWriter.fourcc(*'.mp4'), fps/2, (1280,720))

while True:
    ret, frame = cap.read()

    if not ret:
        break
    # cv2.polylines(frame, [np.array(roi_coords, np.int32)], True, (100, 150, 30), 3, lineType=cv2.LINE_AA)

    # Folder where face images stored
    face_images = glob.glob("output/*.jpg")

    # If Counter
    resetCounter.append(detections_counter)
    if len(resetCounter) > 2:
        resetCounter.pop(0)

    # Get the current time in "H:M AM/PM" format Draw Banner on Top of Frame
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")    
    cv2.rectangle(frame, (0, 0), (380, 30), (0,120, 110), -1)  # Rectangle dimensions and color
    # Put the current time on top of the black rectangle
    cv2.putText(frame, f"Attendance Monitoring: {current_time}", (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (190, 215, 255), 1, cv2.LINE_AA)

    # Person Detection
    res = model(frame) # Detection

    for psn in res.xyxy[0]:
        x1, y1, x2, y2,conf,label = map(int, psn[:6])

        # Calculate the center point of the bounding box
        center_x = ((x1 + x2) / 2)
        center_y = y2-20
        center_point = (int(center_x), int(center_y))

        # Define the color of the circle (BGR format)
        circle_color = (0, 120, 0)  # Green color in BGR
        result = cv2.pointPolygonTest(np.array(roi_coords, dtype=np.int32), center_point, False)

        if result > 0:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 120, 255), 1)
            detections_counter += 1
            # if delayCounter == 10 or delayCounter == 49 or delayCounter == 173:
            if delayCounter == 45 or delayCounter == 46:

                # Extract the person's bounding box
                person = frame[y1:y2, x1:x2]
                # Convert the person bounding box to grayscale for face detection
                gray_person = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                # Perform face detection within the person bounding box using Haar Cascade
                faces = face_cascade.detectMultiScale(gray_person, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    # Adjust the face coordinates based on the person's bounding box
                    fx1, fy1, fx2, fy2 = x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh
                    # Extract the face
                    face = frame[fy1-5:fy2+5, fx1-5:fx2+5]
                    # Save the passport-sized photo as a separate image
                    cv2.imwrite(f'output/face.jpg', face)

            if len(face_images) > 0:
                y_offset = 40
                image = cv2.imread(face_images[-1])
                height, width, _ = image.shape
                frame[y_offset:y_offset + height, 40:width + 40] = image
        else:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if len(resetCounter) > 1:
            if resetCounter[1] - resetCounter[0] == 0:
                delayCounter = 0
                try:
                    if len(face_images) > 0:
                        os.remove(face_images[-1])
                except: pass
            else:
                delayCounter += 1

    cv2.imshow("Camera-1", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# out.release()