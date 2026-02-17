import os
import sys
import time
import pickle
import logging
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import cv2
import torch
import requests
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response
from facenet_pytorch import InceptionResnetV1, MTCNN

# Custom credential file
import credentials

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(BASE_DIR, 'security_cam_log.txt')
    ALERTS_DIR = os.path.join(BASE_DIR, 'alerts')
    FACE_VECTORS_FILE = os.path.join(BASE_DIR, 'face_vectors.pkl')
    
    # Models
    YOLO_MODEL = 'yolo26n.pt'
    
    # Security Parameters
    RECOGNITION_THRESHOLD = 0.7
    GRACE_PERIOD = 15 # seconds
    DETECTION_RESET_TIME = 30 # seconds
    SAFE_THRESHOLD = 1 # detections
    SAFE_MODE_DURATION = 45 # seconds
    SECURITY_LOOP_DELAY = 0.1 # seconds
    
    # Buffer Parameters
    BUFFER_DURATION = 20 # seconds
    BUFFER_FPS = 1
    BUFFER_MAX_FRAMES = BUFFER_DURATION * BUFFER_FPS
    ALERT_STORAGE_DURATION = 30 # days
    
    # Camera
    STREAM_URL = credentials.stream_url

    # Telegram Configuration
    TELEGRAM_TOKEN = getattr(credentials, 'telegram_token', None)
    TELEGRAM_CHAT_ID = getattr(credentials, 'telegram_chat_id', None)
    TELEGRAM_COOLDOWN = 300 # seconds between alerts

# Ensure directories exist
os.makedirs(Config.ALERTS_DIR, exist_ok=True)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# -------------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Shared Resources
# -------------------------------------------------------------------------

class SharedState:
    """Thread-safe storage for the latest frame to be served by Flask."""
    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

    def update(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

shared_state = SharedState()
app = Flask(__name__)

# -------------------------------------------------------------------------
# Core Classes
# -------------------------------------------------------------------------

class TelegramNotifier:
    """Handles sending alerts to Telegram."""
    def __init__(self, token: str, chat_id: str, cooldown: int):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.last_alert_time = 0

    def send_alert(self, message: str, frames: List[np.ndarray]):
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not set. Skipping alert.")
            return
        
        now = time.time()
        if now - self.last_alert_time < self.cooldown:
            return

        # Send text message
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"

            data = {'chat_id': self.chat_id, 'text': message}
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully.")
                self.last_alert_time = now
            else:
                logger.error(f"Telegram API Error: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

        # Send video evidence
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_path = temp_video.name
            
            # Compile video using OpenCV
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, Config.BUFFER_FPS, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()

            # Send video to Telegram
            url = f"https://api.telegram.org/bot{self.token}/sendVideo"
            with open(temp_path, 'rb') as video_file:
                files = {'video': video_file}
                data = {'chat_id': self.chat_id, 'caption': 'Alert Video'}
                response = requests.post(url, data=data, files=files, timeout=60)
                if response.status_code == 200:
                    logger.info("Telegram video alert sent successfully.")
                else:
                    logger.error(f"Telegram API Error: {response.text}")
            os.remove(temp_path)
        except Exception as e:
            logger.error(f"Failed to send Telegram video alert: {e}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

class FaceRecognizer:
    """Handles Face Detection (MTCNN) and Recognition (InceptionResnet)."""
    def __init__(self, vectors_path):
        self.mtcnn = MTCNN(
            margin=40,
            thresholds=[0.5, 0.6, 0.6]
        )
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.known_faces = self._load_vectors(vectors_path)

    def _load_vectors(self, path) -> Dict[str, List[np.ndarray]]:
        try:
            with open(path, 'rb') as f:
                faces = pickle.load(f)
                logger.info(f"Loaded known faces: {list(faces.keys())}")
                return faces
        except FileNotFoundError:
            logger.warning("face_vectors.pkl not found. No known faces loaded.")
            return {}

    def identify(self, frame: np.ndarray, box) -> tuple[str, Optional[dict]]:
        """Returns the name of the person or 'Unknown' along with the minimum 
        distance to known faces if a face is detected else None."""
        # Crop the person image
        img_crop = self._crop_person(frame, box)

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        # Detect face
        start_time = time.time()
        face_tensor = self.mtcnn(img_rgb)
        mtcnn_time = time.time() - start_time
        # logger.debug(f"MTCNN detection time: {mtcnn_time:.3f}s")
        if face_tensor is None:
            # logger.debug("No face detected by MTCNN.")
            return "Unknown", {'No Face Detected': None}

        # Generate embedding
        with torch.no_grad():
            start_time = time.time()
            embedding = self.facenet(face_tensor.unsqueeze(0)).detach().numpy()
            facenet_time = time.time() - start_time
            # logger.debug(f"Facenet embedding time: {facenet_time:.3f}s")

        # Compare with known faces
        min_dist_dict = {}
        for name, vectors in self.known_faces.items():
            min_dist_dict[name] = min(np.linalg.norm(embedding - known_vec) for known_vec in vectors)
            if min_dist_dict[name] < Config.RECOGNITION_THRESHOLD:
                return name, min_dist_dict[name]
        # logger.debug("Face detected but no match.")
        return "Unknown", min_dist_dict
    
    def _crop_person(self, frame, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxw = x2 - x1
        boxh = y2 - y1

        # Adjust to square crop
        if boxh > boxw:
            side = boxw
            y2 = y1 + side
        else:
            side = boxh
            cx = x1 + boxw // 2
            x1 = cx - side // 2
            x2 = cx + side // 2

        return frame[y1:y2, x1:x2]   


class PersonDetector:
    """Handles Person Detection using YOLO."""
    def __init__(self, model_path):
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

    def detect_persons(self, image: np.ndarray):
        """Returns a YOLO result object containing only person detections."""
        start_time = time.time()
        results = self.model.predict(
            image, 
            conf=0.35, 
            classes=[0], # Class 0 is 'person'
            imgsz=320,
            verbose=False
        )
        yolo_time = time.time() - start_time
        # logger.debug(f"YOLO detection time: {yolo_time:.3f}s")
        return results[0]


class SecurityStateMachine:
    """Tracks detection state to manage alerts and safe modes."""
    def __init__(self, safe_identities: List[str]):
        self.safe_identities = safe_identities
        self.detections = {}
        self.first_sight_time = 0
        self.last_sight_time = 0
        self.safe_mode_start = 0
        self.alerted = False
        self.reset()

    def reset(self):
        self.detections = {name: 0 for name in self.safe_identities + ["Unknown"]}
        self.first_sight_time = 0
        self.last_sight_time = 0
        self.alerted = False

    def update(self, identities: List[str]) -> bool:
        """Updates state based on current frame identities. Returns True if Alert needed."""
        now = time.time()

        # 1. Check Safe Mode
        if now - self.safe_mode_start < Config.SAFE_MODE_DURATION:
            return False

        # 2. Reset if room empty for too long
        if self.last_sight_time > 0 and (now - self.last_sight_time > Config.DETECTION_RESET_TIME):
            logger.info(f"Room empty for {Config.DETECTION_RESET_TIME}s. Resetting state.")
            self.reset()

        # 3. Update Timings & Counts
        if identities:
            self.last_sight_time = now
            if self.first_sight_time == 0:
                self.first_sight_time = now

            for identity in identities:
                self.detections[identity] = self.detections.get(identity, 0) + 1

        # 4. Check for Safe Identities
        for name in self.safe_identities:
            if self.detections.get(name, 0) >= Config.SAFE_THRESHOLD:
                logger.info(f"Safe identity '{name}' confirmed. Entering Safe Mode.")
                self.safe_mode_start = now
                self.reset()
                return False

        # 5. Grace Period
        if (now - self.first_sight_time) < Config.GRACE_PERIOD:
            return False

        # 6. Intruder Alert Logic
        # (Simple logic: if we are past grace period and haven't triggered safe mode yet)
        most_likely = max(self.detections, key=self.detections.get)
        if most_likely == "Unknown" and self.detections["Unknown"] >= 0:
            if not self.alerted:
                logger.warning("!!! Intruder Alert: Unknown Person Detected !!!")
                self.alerted = True
            return True

        return False


class CameraReader:
    """Reads frames from a stream in a separate thread to prevent blocking."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video stream at {url}")
        
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# -------------------------------------------------------------------------
# Application Logic
# -------------------------------------------------------------------------

class SecuritySystem:
    def __init__(self, safe_identities):
        self.camera = CameraReader(Config.STREAM_URL)
        self.detector = PersonDetector(Config.YOLO_MODEL)
        self.face_rec = FaceRecognizer(Config.FACE_VECTORS_FILE)
        self.state_machine = SecurityStateMachine(safe_identities)
        
        # Telegram Notifier
        self.notifier = TelegramNotifier(
            Config.TELEGRAM_TOKEN, 
            Config.TELEGRAM_CHAT_ID, 
            Config.TELEGRAM_COOLDOWN
        )

        # Evidence Buffer
        self.frame_buffer = deque(maxlen=Config.BUFFER_MAX_FRAMES)
        self.last_buffer_time = 0
        self.last_buffer_dump = 0

    def run(self):
        logger.info("Security Feed Started.")
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # Process Frame
                annotated_frame = self._process_frame(frame)
                
                # Update Shared State for Web View
                shared_state.update(annotated_frame)
                
                # Update Evidence Buffer
                self._update_buffer(annotated_frame)

                time.sleep(Config.SECURITY_LOOP_DELAY) # Small sleep to prevent overheating

        except KeyboardInterrupt:
            logger.info("Stopping security feed...")
        finally:
            self.camera.stop()

    def _process_frame(self, frame):
        # Detect
        result = self.detector.detect_persons(frame)
        annotated_frame = result.plot(labels=False)


        current_identities = []
        distance_dicts = []

        if result.boxes:
            # Identify
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                identity, min_dist_dict = self.face_rec.identify(frame, box)

                current_identities.append(identity)
                distance_dicts.append(min_dist_dict)

                # Draw label
                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                cv2.putText(annotated_frame, identity, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            logger.info(f"Detected: {current_identities}")
            for identity, min_dist_dict in zip(current_identities, distance_dicts):
                logger.debug(f'Detected: {identity} | Distances: {min_dist_dict}')

        # Update Security State
        alert_triggered = self.state_machine.update(current_identities)
        if alert_triggered:
            self._save_evidence()
            self.notifier.send_alert(
                "Intruder Alert: Unknown Person Detected!", 
                list(self.frame_buffer)
            )
            
        return annotated_frame         

    def _update_buffer(self, frame):
        now = time.time()
        if now - self.last_buffer_time >= 1.0 / Config.BUFFER_FPS:
            self.frame_buffer.append(frame.copy())
            self.last_buffer_time = now

    def _save_evidence(self):
        # Throttle saves
        if time.time() - self.last_buffer_dump < Config.BUFFER_DURATION + 5:
            return

        logger.info("Saving evidence snapshot...")
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        save_dir = os.path.join(Config.ALERTS_DIR, f"intrusion_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        snapshot = list(self.frame_buffer)
        for i, frame in enumerate(snapshot):
            cv2.imwrite(os.path.join(save_dir, f"{i:03d}.jpg"), frame)
        
        self.last_buffer_dump = time.time()

        # Cleanup old evidence
        self._cleanup_evidence()

    def _cleanup_evidence(self):
        now = time.time()
        for folder in os.listdir(Config.ALERTS_DIR):
            folder_path = os.path.join(Config.ALERTS_DIR, folder)
            if os.path.isdir(folder_path):
                folder_time = os.path.getmtime(folder_path)
                if now - folder_time > Config.ALERT_STORAGE_DURATION * 86400:
                    logger.info(f"Deleting old evidence: {folder_path}")
                    try:
                        for file in os.listdir(folder_path):
                            os.remove(os.path.join(folder_path, file))
                        os.rmdir(folder_path)
                    except Exception as e:
                        logger.error(f"Error deleting evidence folder: {e}")


# -------------------------------------------------------------------------
# Flask Routes
# -------------------------------------------------------------------------

def generate_feed():
    while True:
        frame = shared_state.get()
        if frame is None:
            time.sleep(0.1)
            continue
            
        flag, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if flag:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encoded) + b'\r\n')
            
        time.sleep(1.0)

@app.route('/')
def index():
    return """
        <html>
        <head>
          <title>Security Feed</title>
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <style>
            html, body {
              margin: 0;
              padding: 0;
              height: 100%;
              background: black;
            }
            body {
              display: flex;
              justify-content: center;
              align-items: center;
            }
            img {
              width: 100vw;
              height: 100vh;
              object-fit: contain;  /* keep aspect ratio, no cropping */
            }
          </style>
        </head>
        <body>
          <img src="/video_feed">
        </body>
        </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Initializing Security System...")
    
    # 1. Get Safe Identities
    user_input = input("Enter safe identities (comma-separated): ")
    safe_ids = [n.strip() for n in user_input.split(',') if n.strip()]
    
    # 2. Start Security System in a Background Thread
    system = SecuritySystem(safe_identities=safe_ids)
    security_thread = threading.Thread(target=system.run, daemon=True)
    security_thread.start()

    # 3. Start Web Server
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask runtime error: {e}")
