import cv2
from modules.handTrackingModule import HandDetect, FrameRateDisplay

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