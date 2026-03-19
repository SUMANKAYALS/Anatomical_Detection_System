import cv2
import mediapipe as mp
import numpy as np
import math
import time

class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode        = mode
        self.maxHands    = maxHands
        self.detectionCon = detectionCon
        self.trackCon    = trackCon

        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHand(self, frame, draw=True, flipType=False):
        imgRGB    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands  = []
        h, w, _   = frame.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(
                self.results.multi_handedness,
                self.results.multi_hand_landmarks
            ):
                myHand  = {}
                lmlist  = []
                xlist, ylist = [], []

                for id, lm in enumerate(handLms.landmark):
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pz = int(lm.z * w)
                    lmlist.append([px, py, pz])
                    xlist.append(px)
                    ylist.append(py)

                xmin, xmax = min(xlist), max(xlist)
                ymin, ymax = min(ylist), max(ylist)
                boxW = xmax - xmin
                boxH = ymax - ymin
                bbox = (xmin, ymin, boxW, boxH)
                cx   = xmin + boxW // 2
                cy   = ymin + boxH // 2

                myHand["lmlist"]  = lmlist
                myHand["bbox"]    = bbox
                myHand["center"]  = (cx, cy)
                myHand["type"]    = (
                    "Left"  if handType.classification[0].label == "Right"
                    else "Right"
                ) if flipType else handType.classification[0].label

                allHands.append(myHand)

                if draw:
                    color = (0, 200, 255) if myHand["type"] == "Right" else (255, 100, 0)
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        self.mpDraw.DrawingSpec(color=(200, 200, 200), thickness=1)
                    )
                    pad = 18
                    cv2.rectangle(
                        frame,
                        (bbox[0] - pad, bbox[1] - pad),
                        (bbox[0] + bbox[2] + pad, bbox[1] + bbox[3] + pad),
                        color, 2
                    )
                    label = f"{myHand['type']} Hand"
                    cv2.putText(frame, label,
                                (bbox[0] - pad, bbox[1] - pad - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return (allHands, frame) if draw else allHands

    def fingersUp(self, myHand):
        fingers   = []
        handType  = myHand["type"]
        lmlist    = myHand["lmlist"]
        fingers.append(1 if (
            lmlist[4][0] > lmlist[3][0] if handType == "Right"
            else lmlist[4][0] < lmlist[3][0]
        ) else 0)
        for i in range(1, 5):
            fingers.append(
                1 if lmlist[self.tipIds[i]][1] < lmlist[self.tipIds[i] - 2][1] else 0
            )
        return fingers

    def findDistance(self, p1, p2, frame=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        if frame is not None:
            cv2.circle(frame, (x1, y1), 8, (255, 0, 200), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 200), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 200), 2)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), cv2.FILLED)
            return length, (x1, y1, x2, y2, cx, cy), frame
        return length, (x1, y1, x2, y2, cx, cy)


# ────────────────────────────────────────────────
#  DRAWING HELPERS
# ────────────────────────────────────────────────

def draw_label(frame, text, pos, color=(255, 255, 255), scale=0.55, thickness=1):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + 4),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_bone(frame, p1, p2, color, thickness=2):
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_joint(frame, point, color, radius=6):
    cv2.circle(frame, point, radius, color, cv2.FILLED)
    cv2.circle(frame, point, radius + 2, (255, 255, 255), 1)


def px_dist(p1, p2):
    return int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


# ────────────────────────────────────────────────
#  ANATOMICAL REGIONS
# ────────────────────────────────────────────────

# MediaPipe Pose landmark indices
NOSE         = 0
LEFT_EYE     = 2
RIGHT_EYE    = 5
LEFT_EAR     = 7
RIGHT_EAR    = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
LEFT_FOOT      = 31
RIGHT_FOOT     = 32

# Color palette per region
C_HEAD      = (0,   255, 255)   # cyan
C_NECK      = (180, 255, 180)   # light green
C_SHOULDER  = (255, 200,   0)   # gold
C_ARM_R     = (0,   200, 255)   # sky blue
C_ARM_L     = (255, 100,   0)   # orange
C_SPINE     = (200, 100, 255)   # purple
C_HIP       = (255, 255,   0)   # yellow
C_LEG_R     = (0,   255, 120)   # green
C_LEG_L     = (80,  150, 255)   # blue
C_FOOT      = (255,  80, 180)   # pink


def lm_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def detect_anatomy(frame, pose_results, face_results):
    """Draw all anatomical regions with labels and measurements."""
    h, w, _ = frame.shape
    data     = {}

    if not pose_results.pose_landmarks:
        return data

    lm = pose_results.pose_landmarks.landmark

    def pt(idx):
        return lm_pt(lm, idx, w, h)

    # ── collect key points ──
    nose     = pt(NOSE)
    l_eye    = pt(LEFT_EYE)
    r_eye    = pt(RIGHT_EYE)
    l_ear    = pt(LEFT_EAR)
    r_ear    = pt(RIGHT_EAR)
    l_sho    = pt(LEFT_SHOULDER)
    r_sho    = pt(RIGHT_SHOULDER)
    l_elb    = pt(LEFT_ELBOW)
    r_elb    = pt(RIGHT_ELBOW)
    l_wri    = pt(LEFT_WRIST)
    r_wri    = pt(RIGHT_WRIST)
    l_hip    = pt(LEFT_HIP)
    r_hip    = pt(RIGHT_HIP)
    l_kne    = pt(LEFT_KNEE)
    r_kne    = pt(RIGHT_KNEE)
    l_ank    = pt(LEFT_ANKLE)
    r_ank    = pt(RIGHT_ANKLE)

    # Spine mid-points
    neck_mid = ((l_sho[0] + r_sho[0]) // 2, (l_sho[1] + r_sho[1]) // 2)
    hip_mid  = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

    # ════════════════════════════
    #  1. HEAD
    # ════════════════════════════
    cv2.circle(frame, nose, 12, C_HEAD, cv2.FILLED)
    cv2.circle(frame, nose, 14, C_HEAD, 1)
    draw_joint(frame, l_eye, C_HEAD, 4)
    draw_joint(frame, r_eye, C_HEAD, 4)
    draw_joint(frame, l_ear, C_HEAD, 4)
    draw_joint(frame, r_ear, C_HEAD, 4)
    draw_bone(frame, l_ear, r_ear, C_HEAD, 1)
    draw_bone(frame, l_eye, r_eye, C_HEAD, 1)
    head_w = px_dist(l_ear, r_ear)
    draw_label(frame, f"HEAD  W:{head_w}px",
               (nose[0] + 16, nose[1] - 10), C_HEAD)
    draw_label(frame, f"X:{nose[0]} Y:{nose[1]}",
               (nose[0] + 16, nose[1] + 14), C_HEAD, 0.45)
    data["head"] = nose

    # ════════════════════════════
    #  2. NECK
    # ════════════════════════════
    draw_bone(frame, nose, neck_mid, C_NECK, 2)
    draw_joint(frame, neck_mid, C_NECK, 5)
    neck_len = px_dist(nose, neck_mid)
    draw_label(frame, f"NECK  {neck_len}px",
               (neck_mid[0] + 10, neck_mid[1]), C_NECK)

    # ════════════════════════════
    #  3. SHOULDERS
    # ════════════════════════════
    draw_bone(frame, l_sho, r_sho, C_SHOULDER, 3)
    draw_joint(frame, l_sho, C_SHOULDER, 7)
    draw_joint(frame, r_sho, C_SHOULDER, 7)
    sho_w = px_dist(l_sho, r_sho)
    draw_label(frame, f"SHOULDERS  W:{sho_w}px",
               (neck_mid[0] - 60, neck_mid[1] + 18), C_SHOULDER)
    draw_label(frame, "R.Shoulder", (r_sho[0] - 20, r_sho[1] - 14), C_SHOULDER, 0.45)
    draw_label(frame, "L.Shoulder", (l_sho[0] + 5,  l_sho[1] - 14), C_SHOULDER, 0.45)

    # ════════════════════════════
    #  4. RIGHT ARM  (appears on LEFT of mirrored frame)
    # ════════════════════════════
    draw_bone(frame, r_sho, r_elb, C_ARM_R, 3)
    draw_bone(frame, r_elb, r_wri, C_ARM_R, 3)
    draw_joint(frame, r_elb, C_ARM_R, 6)
    draw_joint(frame, r_wri, C_ARM_R, 6)
    upper_r = px_dist(r_sho, r_elb)
    lower_r = px_dist(r_elb, r_wri)
    draw_label(frame, f"R.Arm  U:{upper_r} L:{lower_r}px",
               (r_elb[0] - 10, r_elb[1] + 20), C_ARM_R, 0.48)

    # ════════════════════════════
    #  5. LEFT ARM
    # ════════════════════════════
    draw_bone(frame, l_sho, l_elb, C_ARM_L, 3)
    draw_bone(frame, l_elb, l_wri, C_ARM_L, 3)
    draw_joint(frame, l_elb, C_ARM_L, 6)
    draw_joint(frame, l_wri, C_ARM_L, 6)
    upper_l = px_dist(l_sho, l_elb)
    lower_l = px_dist(l_elb, l_wri)
    draw_label(frame, f"L.Arm  U:{upper_l} L:{lower_l}px",
               (l_elb[0] + 10, l_elb[1] + 20), C_ARM_L, 0.48)

    # ════════════════════════════
    #  6. SPINE
    # ════════════════════════════
    draw_bone(frame, neck_mid, hip_mid, C_SPINE, 3)
    draw_joint(frame, hip_mid, C_SPINE, 6)
    spine_len = px_dist(neck_mid, hip_mid)
    draw_label(frame, f"SPINE  {spine_len}px",
               (hip_mid[0] + 10, (neck_mid[1] + hip_mid[1]) // 2), C_SPINE)

    # ════════════════════════════
    #  7. HIPS
    # ════════════════════════════
    draw_bone(frame, l_hip, r_hip, C_HIP, 3)
    draw_joint(frame, l_hip, C_HIP, 7)
    draw_joint(frame, r_hip, C_HIP, 7)
    hip_w = px_dist(l_hip, r_hip)
    draw_label(frame, f"HIPS  W:{hip_w}px",
               (hip_mid[0] - 40, hip_mid[1] + 20), C_HIP)
    draw_label(frame, "R.Hip", (r_hip[0] - 10, r_hip[1] + 18), C_HIP, 0.45)
    draw_label(frame, "L.Hip", (l_hip[0] + 5,  l_hip[1] + 18), C_HIP, 0.45)

    # ════════════════════════════
    #  8. RIGHT LEG
    # ════════════════════════════
    draw_bone(frame, r_hip, r_kne, C_LEG_R, 3)
    draw_bone(frame, r_kne, r_ank, C_LEG_R, 3)
    draw_joint(frame, r_kne, C_LEG_R, 6)
    draw_joint(frame, r_ank, C_LEG_R, 6)
    thigh_r = px_dist(r_hip, r_kne)
    shin_r  = px_dist(r_kne, r_ank)
    draw_label(frame, f"R.Leg  T:{thigh_r} S:{shin_r}px",
               (r_kne[0] - 10, r_kne[1] + 20), C_LEG_R, 0.48)

    # ════════════════════════════
    #  9. LEFT LEG
    # ════════════════════════════
    draw_bone(frame, l_hip, l_kne, C_LEG_L, 3)
    draw_bone(frame, l_kne, l_ank, C_LEG_L, 3)
    draw_joint(frame, l_kne, C_LEG_L, 6)
    draw_joint(frame, l_ank, C_LEG_L, 6)
    thigh_l = px_dist(l_hip, l_kne)
    shin_l  = px_dist(l_kne, l_ank)
    draw_label(frame, f"L.Leg  T:{thigh_l} S:{shin_l}px",
               (l_kne[0] + 10, l_kne[1] + 20), C_LEG_L, 0.48)

    # ════════════════════════════
    #  10. FACE MESH (optional)
    # ════════════════════════════
    if face_results and face_results.multi_face_landmarks:
        mp_draw = mp.solutions.drawing_utils
        mp_draw_styles = mp.solutions.drawing_styles
        mp_face_mesh   = mp.solutions.face_mesh
        for face_lms in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame, face_lms,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 200, 150), thickness=1, circle_radius=1
                )
            )

    data.update({
        "neck_mid": neck_mid, "hip_mid": hip_mid,
        "l_sho": l_sho, "r_sho": r_sho,
        "l_elb": l_elb, "r_elb": r_elb,
        "l_wri": l_wri, "r_wri": r_wri,
        "l_hip": l_hip, "r_hip": r_hip,
        "l_kne": l_kne, "r_kne": r_kne,
        "l_ank": l_ank, "r_ank": r_ank,
    })
    return data


def draw_hud(frame, fps, hands):
    """Overlay HUD panel — FPS, detected parts count, hand info."""
    h, w, _ = frame.shape

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    title = "ANATOMICAL DETECTION SYSTEM"
    cv2.putText(frame, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2, cv2.LINE_AA)

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 100), 2, cv2.LINE_AA)

    # Bottom-left legend
    legend = [
        ("HEAD",      C_HEAD),
        ("NECK",      (180, 255, 180)),
        ("SHOULDERS", C_SHOULDER),
        ("R.ARM",     C_ARM_R),
        ("L.ARM",     C_ARM_L),
        ("SPINE",     C_SPINE),
        ("HIPS",      C_HIP),
        ("R.LEG",     C_LEG_R),
        ("L.LEG",     C_LEG_L),
    ]
    by = h - 10 - len(legend) * 18
    for i, (name, color) in enumerate(legend):
        y = by + i * 18
        cv2.circle(frame, (14, y), 5, color, cv2.FILLED)
        cv2.putText(frame, name, (24, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Hand count bottom-right
    hand_info = f"Hands: {len(hands)}"
    for i, hand in enumerate(hands):
        fingers = sum(hand.get("fingers", [0, 0, 0, 0, 0]))
        hand_info += f"  |  {hand['type']}: {fingers} fingers"
    cv2.putText(frame, hand_info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────

def main():
    cap      = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Pose
    mpPose = mp.solutions.pose
    pose   = mpPose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Face Mesh
    mpFace    = mp.solutions.face_mesh
    face_mesh = mpFace.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow("Anatomical Detection System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Anatomical Detection System", 1280, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Process ──
        pose_results = pose.process(rgb)
        face_results = face_mesh.process(rgb)

        # ── Anatomical skeleton ──
        detect_anatomy(frame, pose_results, face_results)

        # ── Hand detection ──
        hands, frame = detector.findHand(frame, draw=True, flipType=False)

        # Attach finger state to each hand dict
        for hand in hands:
            hand["fingers"] = detector.fingersUp(hand)

        # Finger distance (index tip ↔ thumb tip) for first hand
        if hands:
            lm = hands[0]["lmlist"]
            length, _, frame = detector.findDistance(
                (lm[8][0], lm[8][1]),
                (lm[4][0], lm[4][1]),
                frame
            )
            h_frame, w_frame, _ = frame.shape
            draw_label(frame, f"Finger dist: {int(length)}px",
                       (10, 70), (255, 0, 200))

        # ── FPS ──
        now  = time.time()
        fps  = 1 / (now - prev_time + 1e-9)
        prev_time = now

        # ── HUD overlay ──
        draw_hud(frame, fps, hands)

        cv2.imshow("Anatomical Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



# import cv2
# import mediapipe as mp
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
#                                          self.results.multi_hand_landmarks):

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

#                     # Display hand bounding box dimensions
#                     dim_text = f"W:{boxW} H:{boxH}"
#                     cv2.putText(frame, dim_text,
#                                 (bbox[0] - 20, bbox[1] + bbox[3] + 40),
#                                 cv2.FONT_HERSHEY_PLAIN,
#                                 1.5, (255, 0, 255), 2)

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

#         nose = landmarks[0]
#         x_head, y_head = int(nose.x * w), int(nose.y * h)

#         # Head circle
#         cv2.circle(frame, (x_head, y_head), 10, (0, 255, 255), cv2.FILLED)
#         cv2.putText(frame, "Head", (x_head + 15, y_head),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#         # Display head position dimensions
#         cv2.putText(frame, f"X:{x_head} Y:{y_head}",
#                     (x_head + 15, y_head + 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#         return (x_head, y_head)

#     return None


# def track_body(frame, poseResults):
#     h, w, _ = frame.shape

#     if poseResults.pose_landmarks:
#         landmarks = poseResults.pose_landmarks.landmark

#         # ── LEFT LEG landmarks (appear on RIGHT of flipped frame) ──
#         left_hip   = landmarks[23]
#         left_knee  = landmarks[25]
#         left_ankle = landmarks[27]

#         x1, y1 = int(left_hip.x * w),   int(left_hip.y * h)
#         x2, y2 = int(left_knee.x * w),  int(left_knee.y * h)
#         x3, y3 = int(left_ankle.x * w), int(left_ankle.y * h)

#         # ── RIGHT LEG landmarks (appear on LEFT of flipped frame) ──
#         right_hip   = landmarks[24]
#         right_knee  = landmarks[26]
#         right_ankle = landmarks[28]

#         x4, y4 = int(right_hip.x * w),   int(right_hip.y * h)
#         x5, y5 = int(right_knee.x * w),  int(right_knee.y * h)
#         x6, y6 = int(right_ankle.x * w), int(right_ankle.y * h)

#         # ── Draw RIGHT LEG (landmarks 23,25,27 — swapped due to flip) ──
#         cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
#         cv2.line(frame, (x2, y2), (x3, y3), (255, 255, 0), 3)
#         cv2.putText(frame, "Right Leg", (x2 + 10, y2),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#         # Dimensions for right leg
#         right_leg_len = int(math.hypot(x2 - x1, y2 - y1) +
#                             math.hypot(x3 - x2, y3 - y2))
#         cv2.putText(frame, f"Len:{right_leg_len}px", (x2 + 10, y2 + 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

#         # ── Draw LEFT LEG (landmarks 24,26,28 — swapped due to flip) ──
#         cv2.line(frame, (x4, y4), (x5, y5), (0, 255, 255), 3)
#         cv2.line(frame, (x5, y5), (x6, y6), (0, 255, 255), 3)
#         cv2.putText(frame, "Left Leg", (x5 + 10, y5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         # Dimensions for left leg
#         left_leg_len = int(math.hypot(x5 - x4, y5 - y4) +
#                            math.hypot(x6 - x5, y6 - y5))
#         cv2.putText(frame, f"Len:{left_leg_len}px", (x5 + 10, y5 + 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#         return {
#             "right_leg": [(x1, y1), (x2, y2), (x3, y3)],
#             "left_leg":  [(x4, y4), (x5, y5), (x6, y6)]
#         }

#     return None


# def main():
#     cap      = cv2.VideoCapture(0)
#     detector = HandDetector(detectionCon=0.8, maxHands=2)

#     mpPose = mp.solutions.pose
#     pose   = mpPose.Pose()

#     cv2.namedWindow("AI System", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("AI System", 1280, 720)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape

#         # Display frame dimensions top-left
#         cv2.putText(frame, f"Frame: {w}x{h}",
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (200, 200, 200), 2)

#         # ===== HAND DETECTION =====
#         # flipType=False because frame is already flipped
#         hands, frame = detector.findHand(frame, flipType=False)

#         if hands:
#             hand1  = hands[0]
#             lmlist = hand1["lmlist"]

#             x1, y1 = lmlist[8][0:2]
#             x2, y2 = lmlist[4][0:2]

#             length, _, frame = detector.findDistance((x1, y1), (x2, y2), frame)

#             # Display finger distance
#             cv2.putText(frame, f"Dist:{int(length)}px",
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7, (255, 0, 255), 2)

#         # ===== POSE DETECTION =====
#         rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         poseResults = pose.process(rgb)

#         track_head(frame, poseResults)
#         track_body(frame, poseResults)

#         cv2.imshow("AI System", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()