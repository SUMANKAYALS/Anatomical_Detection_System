# 🧠 BodySense AI — Full Body & Object Tracking System

> A real-time **full-body tracking system** that detects head position, left/right hand classification, leg movement, and object distance — built with Python, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-00C853?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Latest-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F6C90E?style=for-the-badge)

---

## 📌 About

**BodySense AI** is an advanced computer vision system that performs real-time full-body analysis using a standard webcam. Powered by Google's **MediaPipe** and **OpenCV**, it simultaneously tracks multiple body regions and measures object proximity — making it suitable for both **medical** and **security** applications.

### 🔍 What It Tracks

| Body Part | Detection Capability |
|---|---|
| 👤 **Head** | Position, orientation & movement |
| 🖐️ **Hands** | Left/right classification + 21 landmarks each |
| 🦵 **Legs** | Lower body pose & movement tracking |
| 📏 **Objects** | Real-time distance estimation from camera |

---

## 🏥 Medical Applications

BodySense AI serves as a powerful **non-contact detector** in medical environments:

- 🩺 **Patient Movement Monitoring** — Tracks body posture and detects abnormal movements in real time
- 🧍 **Fall Detection** — Identifies sudden leg collapse or loss of body balance
- 💉 **Contactless Assessment** — Enables gesture-based interaction in sterile environments
- 🧠 **Rehabilitation Tracking** — Monitors limb movement recovery progress
- 🚨 **ICU / Ward Surveillance** — Alerts staff when patients leave a defined zone

---

## 🔒 Security Applications

BodySense AI acts as an intelligent **security detector** in surveillance systems:

- 🕵️ **Intruder Detection** — Identifies unauthorized body presence in restricted zones
- 🚧 **Perimeter Monitoring** — Tracks human movement and proximity to boundaries
- ✋ **Gesture-Based Access Control** — Recognize authorized hand signals for door/system unlock
- 📡 **Object Proximity Alerts** — Detects when an object gets dangerously close to a sensor area
- 🎥 **Touchless Security Kiosks** — Hands-free identity verification systems

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture, frame rendering & drawing |
| `mediapipe` | Full-body pose + hand landmark detection |
| `numpy` | Distance calculations & array operations |
| `PyAutoGUI` | Gesture-based system automation |

---

## ⚙️ Installation & Setup

### Step 1 — Install Python & a Code Editor

- Download Python **(3.8 or higher)** from [python.org](https://www.python.org/downloads/)
- ⚠️ On Windows, check **"Add Python to PATH"** during setup
- Install [VS Code](https://code.visualstudio.com/) or any editor you prefer

---

### Step 2 — Install Required Libraries

Open your **Terminal** (macOS/Linux) or **PowerShell** (Windows) and run:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install PyAutoGUI
```

---

### Step 3 — Run the Project

Open the project in your code editor, paste the source code, and save it as `BodySenseAI.py`. Then run:

```bash
python BodySenseAI.py
```

Or press **`F5`** directly in VS Code.

---

### Step 4 — You're All Set! 🎉

Your webcam will activate and BodySense AI will begin tracking your head, hands (left/right), legs, and nearby object distances in real time. **Well done!**

---

## 📁 Project Structure

```
BodySenseAI/
│
├── BodySenseAI.py       # Core full-body tracking logic
└── README.md            # Project documentation
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Suman Kayal**
- GitHub: [@suman_kayal](https://github.com/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by Suman Kayal &nbsp;|&nbsp; BodySense AI — Seeing the Human Body, Intelligently.</p>