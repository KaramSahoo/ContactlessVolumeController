import cv2
import numpy as np
import time
import math
import HandTracker as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##########################################################################################

wCam, hCam = 640, 480
pTime = 0
vol = 0
volBar = 400
volPerc = 0
area = 0

##########################################################################################

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
minVolume = volume_range[0]
maxVolume = volume_range[1]
volume.SetMasterVolumeLevel(0, None)

##########################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detection_conf=0.6, maxHands=1)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame, draw=False)
    lmList, bbox = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:

        # Filters
        wB, hB = bbox[2]-bbox[0], bbox[3]-bbox[1]
        area = wB*hB//100
        if area > 200 and area < 1200:

            # Distance between thumb and index
            length, frame, lineCoord = detector.findDistance(4, 8, frame)

            # Convert Volume
            # volBar = np.interp(length, [50, 200], [400, 150])
            volBar = np.interp(length, [50, 200], [550, 140])
            volPerc = np.interp(length, [50, 200], [0, 100])
            
            # Make stops for volume values
            smoothness = 10
            volPerc = smoothness*round(volPerc/smoothness)

            # If pinky down set volume
            fingers = detector.fingersUp()
            # print(fingers)
            if not fingers[3]:
                volume.SetMasterVolumeLevelScalar(volPerc/100, None)
                cv2.circle(frame, (lineCoord[4], lineCoord[5]), 8, (0, 255, 0), cv2.FILLED)

    # Drawings
        #Vertical Bar
    # cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    # cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        #Horizontal Bar
    cv2.rectangle(frame, (140, 420), (550, 450), (255, 255, 255), 1)
    cv2.rectangle(frame, (int(volBar), 420), (550, 450), (255, 255, 255), cv2.FILLED)
    
    cv2.putText(frame, f'{int(volPerc)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cVol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(frame, f'Volume Set: {int(cVol)}', (400, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)