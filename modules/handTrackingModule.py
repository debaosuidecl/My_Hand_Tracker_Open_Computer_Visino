import cv2
import mediapipe as mp
import time


class FrameRateDisplay():

    def __init__(self):
        self.pTime = 0
        self.cTime = 0

    def showFPS(self, img, draw=True):
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        if draw:
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,
                        3, (255, 0, 255), 3)
        return img


class HandDetect():

    def __init__(self,
                 mode=False,
                 maxHands=2,
                 detectionConfidence=0.5,
                 trackConfidence=0.5,
                 modelComplexity=1):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.modelComplexity,
                                        self.detectionConfidence,
                                        self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findNodePositionsOfHand(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            if myHand:
                for id, lm in enumerate(myHand.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    #height width channel
                    cx, cy = int(lm.x * w), int(
                        lm.y * h
                    )  #obtaining the pixel value of pointer positions in hand
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255),
                                   cv2.FILLED)
        return lmList


def main():

    cap = cv2.VideoCapture(1)
    detector = HandDetect()
    frameRate = FrameRateDisplay()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findNodePositionsOfHand(img, 0, False)

        if len(lmList) != 0:
            controlList = [lmList[4], lmList[8]]
            for pt in controlList:
                cv2.circle(img, (pt[1], pt[2]), 25, (24, 0, 198), cv2.FILLED)

        img = frameRate.showFPS(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()