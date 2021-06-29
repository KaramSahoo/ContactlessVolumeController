import cv2
import mediapipe as mp
import time
import math

class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detection_conf = 0.5, track_conf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        self.lmList = []
        bbox = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        
        #thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        #Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, frame, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        xM, yM = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (xM, yM), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        length = math.hypot(x2-x1, y2-y1)
        return length, frame, [x1, y1, x2, y2, xM, yM]


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        _, frame = cap.read()
        draw_lines = True
        frame = detector.findHands(frame, draw=draw_lines)
        draw_circle = True
        lmList = detector.findPosition(frame, draw=draw_circle)

        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()