# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# import math  


# class HandDetector():

#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackcon=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = detectionCon
#         self.trackcon = trackcon

#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(
#             static_image_mode=self.mode,
#             max_num_hands=self.maxHands,
#             min_detection_confidence=self.detectionCon,
#             min_tracking_confidence=self.trackcon
#         )

#         self.mpDraw = mp.solutions.drawing_utils
#         self.tipIds = [4, 8, 12, 16, 20]

#     def findHand(self, frame, draw=True, flipType=True):
#         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)

#         allHands = []
#         h, w, c = frame.shape

#         if self.results.multi_hand_landmarks:
#             for handType, handLms in zip(self.results.multi_handedness,
#                                         self.results.multi_hand_landmarks):

#                 myHand = {}
#                 mylmlist = []
#                 xlist, ylist = [], []

#                 for id, lm in enumerate(handLms.landmark):
#                     px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
#                     mylmlist.append([px, py, pz])
#                     xlist.append(px)
#                     ylist.append(py)

#                 xmin, xmax = min(xlist), max(xlist)
#                 ymin, ymax = min(ylist), max(ylist)

#                 boxW, boxH = xmax - xmin, ymax - ymin
#                 bbox = xmin, ymin, boxW, boxH
#                 cx, cy = xmin + boxW // 2, ymin + boxH // 2

#                 myHand["lmlist"] = mylmlist
#                 myHand["bbox"] = bbox
#                 myHand["center"] = (cx, cy)

#                 if flipType:
#                     if handType.classification[0].label == "Right":
#                         myHand["type"] = "Left"
#                     else:
#                         myHand["type"] = "Right"
#                 else:
#                     myHand["type"] = handType.classification[0].label

#                 allHands.append(myHand)

#                 if draw:
#                     self.mpDraw.draw_landmarks(frame, handLms,
#                                                self.mpHands.HAND_CONNECTIONS)
#                     cv2.rectangle(frame,
#                                   (bbox[0] - 20, bbox[1] - 20),
#                                   (bbox[0] + bbox[2] + 20,
#                                    bbox[1] + bbox[3] + 20),
#                                   (255, 0, 255), 2)
#                     cv2.putText(frame, myHand["type"],
#                                 (bbox[0] - 30, bbox[1] - 30),
#                                 cv2.FONT_HERSHEY_PLAIN,
#                                 2, (255, 0, 255), 2)

#         return (allHands, frame) if draw else allHands

#     def fingersUp(self, myHand):
#         fingers = []
#         myHandType = myHand["type"]
#         mylmlist = myHand["lmlist"]

#         # Thumb
#         if myHandType == "Right":
#             fingers.append(1 if mylmlist[4][0] > mylmlist[3][0] else 0)
#         else:
#             fingers.append(1 if mylmlist[4][0] < mylmlist[3][0] else 0)

#         # Other fingers
#         for id in range(1, 5):
#             if mylmlist[self.tipIds[id]][1] < mylmlist[self.tipIds[id] - 2][1]:
#                 fingers.append(1)
#             else:
#                 fingers.append(0)

#         return fingers

#     def findDistance(self, p1, p2, frame=None):
#         x1, y1 = p1
#         x2, y2 = p2

#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#         length = math.hypot(x2 - x1, y2 - y1)

#         if frame is not None:
#             cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
#             cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
#             cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
#             cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

#             return length, (x1, y1, x2, y2, cx, cy), frame

#         return length, (x1, y1, x2, y2, cx, cy)

# def track_head(frame, poseResults):
#     h, w, _ = frame.shape

#     if poseResults.pose_landmarks:
#         landmarks = poseResults.pose_landmarks.landmark

#         # Nose = Head reference
#         nose = landmarks[0]

#         x_head, y_head = int(nose.x * w), int(nose.y * h)

#         # Draw
#         cv2.circle(frame, (x_head, y_head), 10, (0, 255, 255), cv2.FILLED)
#         cv2.putText(frame, "Head", (x_head, y_head - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#         return (x_head, y_head)

#     return None

# def track_body(frame, poseResults):
#     h, w, _ = frame.shape

#     if poseResults.pose_landmarks:
#         landmarks = poseResults.pose_landmarks.landmark

#         # 🦵 LEFT LEG (hip → knee → ankle)
#         left_hip = landmarks[23]
#         left_knee = landmarks[25]
#         left_ankle = landmarks[27]

#         x1, y1 = int(left_hip.x * w), int(left_hip.y * h)
#         x2, y2 = int(left_knee.x * w), int(left_knee.y * h)
#         x3, y3 = int(left_ankle.x * w), int(left_ankle.y * h)

#         # 🦵 RIGHT LEG
#         right_hip = landmarks[24]
#         right_knee = landmarks[26]
#         right_ankle = landmarks[28]

#         x4, y4 = int(right_hip.x * w), int(right_hip.y * h)
#         x5, y5 = int(right_knee.x * w), int(right_knee.y * h)
#         x6, y6 = int(right_ankle.x * w), int(right_ankle.y * h)

#         # Draw LEFT leg
#         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
#         cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 255), 3)

#         # Draw RIGHT leg
#         cv2.line(frame, (x4, y4), (x5, y5), (255, 255, 0), 3)
#         cv2.line(frame, (x5, y5), (x6, y6), (255, 255, 0), 3)

#         # Labels
#         cv2.putText(frame, "Left Leg", (x2, y2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         cv2.putText(frame, "Right Leg", (x5, y5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#         return {
#             "left_leg": [(x1, y1), (x2, y2), (x3, y3)],
#             "right_leg": [(x4, y4), (x5, y5), (x6, y6)]
#         }

#     return None

# # def main():
# #     cap = cv2.VideoCapture(0)
# #     detector = HandDetector(detectionCon=0.8, maxHands=2)

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         hands, frame = detector.findHand(frame)

# #         if hands:
# #             hand1 = hands[0]
# #             fingers1 = detector.fingersUp(hand1)

# #             if len(hands) == 2:
# #                 hand2 = hands[1]

# #         cv2.imshow("Frame", frame)

# #         # Press 'q' to exit
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()


# # if __name__ == "__main__":
# #     main()


# def main():
#     cap = cv2.VideoCapture(0)
#     detector = HandDetector(detectionCon=0.8, maxHands=2)

#     mpPose = mp.solutions.pose
#     pose = mpPose.Pose()

#     import pyautogui
#     screen_w, screen_h = pyautogui.size()
#     isMouseDown = False
#     cv2.namedWindow("AI System", cv2.WINDOW_NORMAL)  # ← add this before while loop
#     cv2.resizeWindow("AI System", 1280, 720)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape

#         # ===== HAND DETECTION =====
#         hands, frame = detector.findHand(frame)

#         if hands:
#             hand1 = hands[0]
#             lmlist = hand1["lmlist"]

#             x1, y1 = lmlist[8][0:2]
#             x2, y2 = lmlist[4][0:2]

#             screen_x = screen_w / w * x1
#             screen_y = screen_h / h * y1
#             pyautogui.moveTo(screen_x, screen_y)

#             length, _, frame = detector.findDistance((x1, y1), (x2, y2), frame)

#             if length < 40:
#                 if not isMouseDown:
#                     pyautogui.mouseDown()
#                     isMouseDown = True
#                     print("HOLD")
#             else:
#                 if isMouseDown:
#                     pyautogui.mouseUp()
#                     isMouseDown = False
#                     print("RELEASE")

#         # ===== BODY DETECTION =====  ← now inside the loop
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         poseResults = pose.process(rgb)
#         track_body(frame, poseResults)

#         cv2.imshow("AI System", frame)  # ← now inside the loop

#         if cv2.waitKey(1) & 0xFF == ord('q'):  
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()



import cv2
import mediapipe as mp
import numpy as np
import math


class HandDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackcon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHand(self, frame, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        allHands = []
        h, w, c = frame.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness,
                                         self.results.multi_hand_landmarks):

                myHand = {}
                mylmlist = []
                xlist, ylist = [], []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmlist.append([px, py, pz])
                    xlist.append(px)
                    ylist.append(py)

                xmin, xmax = min(xlist), max(xlist)
                ymin, ymax = min(ylist), max(ylist)

                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = xmin + boxW // 2, ymin + boxH // 2

                myHand["lmlist"] = mylmlist
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label

                allHands.append(myHand)

                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(frame,
                                  (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20,
                                   bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(frame, myHand["type"],
                                (bbox[0] - 30, bbox[1] - 30),
                                cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

                    # Display hand bounding box dimensions
                    dim_text = f"W:{boxW} H:{boxH}"
                    cv2.putText(frame, dim_text,
                                (bbox[0] - 20, bbox[1] + bbox[3] + 40),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.5, (255, 0, 255), 2)

        return (allHands, frame) if draw else allHands

    def fingersUp(self, myHand):
        fingers = []
        myHandType = myHand["type"]
        mylmlist = myHand["lmlist"]

        # Thumb
        if myHandType == "Right":
            fingers.append(1 if mylmlist[4][0] > mylmlist[3][0] else 0)
        else:
            fingers.append(1 if mylmlist[4][0] < mylmlist[3][0] else 0)

        # Other fingers
        for id in range(1, 5):
            if mylmlist[self.tipIds[id]][1] < mylmlist[self.tipIds[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, frame=None):
        x1, y1 = p1
        x2, y2 = p2

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if frame is not None:
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            return length, (x1, y1, x2, y2, cx, cy), frame

        return length, (x1, y1, x2, y2, cx, cy)


def track_head(frame, poseResults):
    h, w, _ = frame.shape

    if poseResults.pose_landmarks:
        landmarks = poseResults.pose_landmarks.landmark

        nose = landmarks[0]
        x_head, y_head = int(nose.x * w), int(nose.y * h)

        # Head circle
        cv2.circle(frame, (x_head, y_head), 10, (0, 255, 255), cv2.FILLED)
        cv2.putText(frame, "Head", (x_head + 15, y_head),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display head position dimensions
        cv2.putText(frame, f"X:{x_head} Y:{y_head}",
                    (x_head + 15, y_head + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return (x_head, y_head)

    return None


def track_body(frame, poseResults):
    h, w, _ = frame.shape

    if poseResults.pose_landmarks:
        landmarks = poseResults.pose_landmarks.landmark

        # ── LEFT LEG landmarks (appear on RIGHT of flipped frame) ──
        left_hip   = landmarks[23]
        left_knee  = landmarks[25]
        left_ankle = landmarks[27]

        x1, y1 = int(left_hip.x * w),   int(left_hip.y * h)
        x2, y2 = int(left_knee.x * w),  int(left_knee.y * h)
        x3, y3 = int(left_ankle.x * w), int(left_ankle.y * h)

        # ── RIGHT LEG landmarks (appear on LEFT of flipped frame) ──
        right_hip   = landmarks[24]
        right_knee  = landmarks[26]
        right_ankle = landmarks[28]

        x4, y4 = int(right_hip.x * w),   int(right_hip.y * h)
        x5, y5 = int(right_knee.x * w),  int(right_knee.y * h)
        x6, y6 = int(right_ankle.x * w), int(right_ankle.y * h)

        # ── Draw RIGHT LEG (landmarks 23,25,27 — swapped due to flip) ──
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 0), 3)
        cv2.putText(frame, "Right Leg", (x2 + 10, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Dimensions for right leg
        right_leg_len = int(math.hypot(x2 - x1, y2 - y1) +
                            math.hypot(x3 - x2, y3 - y2))
        cv2.putText(frame, f"Len:{right_leg_len}px", (x2 + 10, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ── Draw LEFT LEG (landmarks 24,26,28 — swapped due to flip) ──
        cv2.line(frame, (x4, y4), (x5, y5), (0, 255, 255), 3)
        cv2.line(frame, (x5, y5), (x6, y6), (0, 255, 255), 3)
        cv2.putText(frame, "Left Leg", (x5 + 10, y5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Dimensions for left leg
        left_leg_len = int(math.hypot(x5 - x4, y5 - y4) +
                           math.hypot(x6 - x5, y6 - y5))
        cv2.putText(frame, f"Len:{left_leg_len}px", (x5 + 10, y5 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return {
            "right_leg": [(x1, y1), (x2, y2), (x3, y3)],
            "left_leg":  [(x4, y4), (x5, y5), (x6, y6)]
        }

    return None


def main():
    cap      = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    mpPose = mp.solutions.pose
    pose   = mpPose.Pose()

    cv2.namedWindow("AI System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI System", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Display frame dimensions top-left
        cv2.putText(frame, f"Frame: {w}x{h}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

        # ===== HAND DETECTION =====
        # flipType=False because frame is already flipped
        hands, frame = detector.findHand(frame, flipType=False)

        if hands:
            hand1  = hands[0]
            lmlist = hand1["lmlist"]

            x1, y1 = lmlist[8][0:2]
            x2, y2 = lmlist[4][0:2]

            length, _, frame = detector.findDistance((x1, y1), (x2, y2), frame)

            # Display finger distance
            cv2.putText(frame, f"Dist:{int(length)}px",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 255), 2)

        # ===== POSE DETECTION =====
        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poseResults = pose.process(rgb)

        track_head(frame, poseResults)
        track_body(frame, poseResults)

        cv2.imshow("AI System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()