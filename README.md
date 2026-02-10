# Retired-Sentinel

Give your old Android phone a new career in private security.

Retired-Sentinel turns any Android device into a local AI surveillance system using Termux. It detects intruders with YOLO26, identifies faces with FaceNet, streams a live web feed via Flask, and sends you video evidence via Telegram when something's amiss—all running entirely on-device with no cloud required.

## 🎬 Demo

| Intruder Alert (Telegram) | Safe Identity Verified |
|:---:|:---:|
| ![Intruder detected](intruder.gif) | ![Safe ID verified](secure_id.gif) |

*Left: Telegram alert with video clip when unknown person detected. Right: Live feed showing positive identification of authorized user (on device video feed doesn't reflect true frame processing speed).*

## ✨ Features

- **👤 Person Detection**: YOLO26n runs locally to detect people in real-time
- **🔐 Face Recognition**: FaceNet embeddings identify authorized vs. unknown individuals
- **📱 Telegram Alerts**: Instant notifications with video evidence (15-second video file) when intruders detected 
- **🧠 Smart State Machine**: Grace periods, safe mode, and configurable thresholds prevent false alarms
- **♻️ E-Waste Solution**: Resurrects old Android phones as dedicated security hardware

## 🏗️ Architecture

**Pipeline**: Camera stream → YOLO person detection → MTCNN face crop → FaceNet embedding match → Security state machine → Telegram alerts & web stream

## 📦 Installation

### Prerequisites

- Android phone with camera (preferably with decent performance)

### IP Webcam Setup

1. Install IP Webcam from Google Play: https://play.google.com/store/apps/details?id=com.pas.webcam

2. Configure IP Webcam settings:
   - Set resolution to 1920x1440 for best performance
   - Turn off battery optimization for IP Webcam
   - Allow IP Webcam to run in the background
   - Allow IP Webcam to access the camera
   - Allow IP Webcam to display over other apps
   - Set login and password for security (optional but recommended)

### Termux Setup

1. Install Termux from F-Droid: https://f-droid.org/packages/com.termux/

2. Install Termux:API from F-Droid: https://f-droid.org/packages/com.termux.api/

3. Grant Termux permissions for storage:
    - Open Termux and run: `termux-setup-storage`
    - Follow prompts to allow access to storage

4. Install core build tools:
    ```bash
    pkg update && pkg upgrade -y
    pkg install -y python cmake make clang git libjpeg-turbo libpng
    ```

5. Install Python dependencies:

    [!NOTE]
    *Installing PyTorch and OpenCV on Termux can be tricky.*

    ```bash
    # specific termux packages
    pkg install -y tur-repo
    pkg install -y x11-repo
    pkg install -y python-torch python-torchvision opencv-python

    # pip packages
    pip install ultralytics facenet-pytorch Flask PyYAML requests --no-deps
    ```

    If you encounter build errors, ensure `clang`, `cmake`, and `libjpeg-turbo` are installed.

6. Clone the Retired-Sentinel repository to the Downloads directory:
    ```bash
    cd ~/storage/downloads
    git clone https://github.com/Feebami/Retired-Sentinel.git
    ```

*This should probably get you there. There were many hurdles getting the dependencies working on Termux, so if steps fail, search the specific error or ask an LL. Getting Flask, PyTorch, OpenCV, FaceNet-PyTorch, and Ultralytics installed is the goal. Try `--no-deps` with pip if you encounter dependency conflicts and see what happens.*

### Configure video stream URL

1. Open `credentials.py` in an editor (e.g., Pydroid 3).

2. Set stream_url to your IP Webcam stream URL (e.g., `http://Login:Password@127.0.0.1:8080/video`).
    - If you set a login and password in IP Webcam, include them in the URL as shown above.
    - If you did not set a login and password, omit that part.

### Configure Telegram Bot (optional, can be done on a separate device)

1. Create a Telegram bot using BotFather and get the API token.

2. Search for your bot by username and start it.

3. Get your Telegram user ID (you can use @userinfobot).

4. Open `credentials.py` and set `telegram_token` and `telegram_chat_id` with your bot token and user ID.

[!WARNING]
Do not commit your `credentials.py` file to any public repositories, as it contains sensitive information (add to .gitignore).

## 🚀 Usage

### First Run Setup

1. Add Safe Faces:

    Place photos of authorized people in faces/Name1/, faces/Name2/.


2. Generate Embeddings:
    Run the vectorization script to process the faces:
    ```bash
    cd ~/storage/downloads/Retired-Sentinel
    python vectorize_faces.py
    ```

[!NOTE]
*The system will work without authorized faces, but intruder alerts will trigger at any person detection.*

### System Startup

1. Start the main application (from the Retired-Sentinel directory):
    ```bash
    python security_cam.py
    ```

    [!NOTE]
    Initial startup may take a few minutes as models load and the system initializes.

3. The system will initialize and then prompt you for "safe identities" which will be used for security verification. "Safe identities" will trigger safe mode, temporarily disabling intruder alerts. Faces vectorized that aren't set as "safe identities" will not trigger intruder alerts, but will not activate safe mode.

### Customization and Configuration

The system uses FaceNet embeddings for face recognition. To register faces of authorized individuals, place the images of an individual in their own folder inside the `faces` directory. For example:

```faces/
├── Alice/
│   ├── alice1.jpg
│   ├── alice2.jpg
│   └── alice3.jpg
├── Bob/
│   ├── bob1.jpg
│   ├── bob2.jpg
│   └── bob3.jpg
```

Running the vectorization script, `vectorize_faces.py`, will produce a .pkl file with the embeddings for each person that the main application will use for identity verification.

### Live Video Stream

On the device running the system, you can access the live video stream with detections at `http://localhost:5000/video_feed` in a web browser. This stream is for local monitoring and will not reflect the true processing speed due to the overhead of streaming.

## 📁 Project Structure

```
Retired-Sentinel/
├── security_cam.py          # Main security system
├── vectorize_faces.py       # Face embedding generator
├── credentials.py           # API tokens and video stream URL (DO NOT COMMIT TO PUBLIC REPO WITH SENSITIVE INFO) 
├── env.yml                  # Conda dependencies (for reference)
├── face_vectors.pkl         # Generated embeddings (created by vectorize_faces.py)
├── faces/                   # Training photos for safe identities
├── alerts/                  # Saved evidence snapshots
├── intruder.gif             # Demo: Alert notification
├── secure_id.gif            # Demo: Safe ID verification
└── test_images/             # Test images for vector validation
```

## ⚙️ Configuration

Key parameters in `security_cam.py` that you may want to adjust:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `Config.RECOGNITION_THRESHOLD` | Distance threshold for face recognition (lower is stricter) | 0.8 |
| `Config.GRACE_PERIOD` | Time in seconds to wait before an intruder alert can trigger from first identification | 15 |
| `Config.DETECTION_RESET_TIME` | Time in seconds to reset detection state if no person is detected | 60 |
| `Config.SAFE_THRESHOLD` | Number of positive IDs of a person required to be considered "safe" | 2 |
| `Config.SAFE_MODE_DURATION` | Time in seconds to stay in safe mode after identifying a safe person | 90 |
| `Config.SECURITY_LOOP_DELAY` | Time in seconds to wait between each loop iteration of the security system (to prevent overheating) | 0.1 |

## 🧠 Model Information

- **YOLO26n**: yolo26n.pt - Person detection at ~320px for speed

- **MTCNN**: Face detection and alignment

- **Inception ResNet V1**: 512-dimensional face embeddings (pretrained on VGGFace2)

Performance on a 2020-era phone: ~1-2 FPS detection loop, sufficient for security monitoring.

## 📝 Notes

- **Power**: Keep the phone plugged in. Running camera + AI continuously drains battery.

- **Heat**: Extended use may cause thermal throttling. Consider a fan or heatsink for 24/7 operation.

- **Privacy**: All processing is on-device. No data is sent to the cloud unless you configure Telegram alerts.

- **Device Compatibility**: This system can easily be adapted to work with a local webcam on a PC or Raspberry Pi (results may vary based on hardware capabilities). Just make sure to adjust the video stream URL (0 if using a local webcam) and ensure the necessary dependencies are installed. 

## 🔒 Security & Privacy

- **End-of-Life Device Repurposing**: Using a phone that no longer receives updates is vulnerable to any exploits discovered after its last update. 
    - **Risk Mitigation**: Factory reset the phone before use, only install necessary apps (IP Webcam and Termux), and do not use the phone for any other activities to minimize attack surface.

- **Flask Video Streaming**: The default configuration streams the camera feed to `localhost`, which prevents external access. This can be changed if you want to view the stream on another device on the same network.

- **IP Webcam**: Needs to stream to LAN for the system to work. Ensure you set a strong login and password in IP Webcam to prevent unauthorized access to the camera stream.

- **PII Handling**: Face images, embeddings, and alert evidence are stored locally on the device. Optionally, you can delete the `faces` folder after vectorization.

## 🛠️ Troubleshooting

### "No camera detected"

Ensure Termux has camera permissions: `termux-camera-photo -c 0 test.jpg`

If using IP camera, verify `stream_url` is accessible from the phone (try opening the URL in a browser on the phone).

### "Face not recognized"

Add more angles/lighting variations to faces/name/

Collect samples for vectorization from the alerts folder that contain the person you want to positively identify.

If known faces aren't recognized, increase the threshold of `Config.RECOGNITION_THRESHOLD` (e.g., 0.8 → 0.9) to be more lenient.

### Out of memory

Reduce imgsz in YOLO from 320 to 240

Use YOLO_MODEL = 'yolo11n.pt' (smallest variant)

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO26
- [Facenet-Pytorch](https://github.com/timesler/facenet-pytorch) for face recognition
- [Termux](https://termux.com/) for enabling Linux tools on Android
- [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) for camera streaming
- [Telegram](https://telegram.org/) for instant notifications