import cv2
import numpy as np
import time
import math
import threading
import os
import csv
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO
import azure.cognitiveservices.speech as speechsdk
import requests
import uuid
import pytz
import logging
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


# ======= Azure Speech SDK Initialization =======
speech_key = os.getenv("AZURE_KEY")
service_region = os.getenv("AZURE_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
recognizer_lock = threading.Lock()

# ======= Setup console logging (no file) =======
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger()
logging.getLogger("picamera2").setLevel(logging.INFO)

# ======= Load Keywords from CSV =======
def load_keywords_from_csv(csv_filepath):
    categories = {}
    with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for category, keywords_str in row.items():
                if category not in categories:
                    categories[category] = set()
                if keywords_str:
                    keywords = {k.strip().lower() for k in keywords_str.split(',') if k.strip()}
                    categories[category].update(keywords)
    return categories

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data/HelpKeywords-HelloKeywords-OkKeywords.csv")
keyword_categories = load_keywords_from_csv(csv_path)

responses = {
    "Help_Keywords": "I'll notify your caretaker, help is on the way.",
    "Hello_Keywords": "Hello! How can I assist you today?",
    "Ok_Keywords": "Glad to hear you're alright.",
}

alert_response_active = False
patient_id = 1

# ======= Backend Alert API Endpoint =======
SERVER_ALERT_URL = "http://sens-able.manipal.net:8000/alerts/alerts"


def get_mac_address():
    mac_int = uuid.getnode()
    mac = ':'.join(['{:02x}'.format((mac_int >> ele) & 0xff) for ele in range(40, -1, -8)])
    return mac


def get_ist_time():
    """Get the current IST time in ISO format."""
    ist_timezone = pytz.timezone('Asia/Kolkata')
    ist_time = datetime.now(ist_timezone).replace(microsecond=0)
    return ist_time.isoformat()


def send_alert(alert_content):
    mac_address = get_mac_address()
    alert_data = {
        "mac_address": mac_address,
        "alert_content": alert_content,
        "time": get_ist_time()
    }

    def _send():
        try:
            resp = requests.post(SERVER_ALERT_URL, json=alert_data)
            if resp.status_code == 200:
                logger.info(f"Alert sent successfully: {alert_content}")
            else:
                logger.info(f"Failed to send alert: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.info(f"Error sending alert: {e}")

    threading.Thread(target=_send, daemon=True).start()

# ======= Speech Interaction Functions =======
def speak(text):
    with recognizer_lock:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Spoken: {text}")


def listen_for_keywords(keyword_categories, timeout=60, ok_response=None, help_response=None, context="activity response"):
    global alert_response_active
    start_time = time.time()
    while time.time() - start_time < timeout:
        logger.info(f"Listening for {context}...")
        with recognizer_lock:
            result = speech_recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            command = result.text.lower()
            logger.info(f"Patient said: {command}")
            for keyword in keyword_categories.get("Ok_Keywords", []):
                if keyword in command:
                    if ok_response:
                        speak(ok_response)
                    alert_response_active = False
                    return "ok"
            for keyword in keyword_categories.get("Help_Keywords", []):
                if keyword in command:
                    if help_response:
                        speak(help_response)
                    send_alert("patient_needs_help")
                    alert_response_active = False
                    return "help"
        elif result.reason == speechsdk.ResultReason.Canceled:
            logger.info("Speech recognition canceled.")
            time.sleep(2)
        else:
            logger.info("No or unrecognized response, listening again...")
            time.sleep(2)
    alert_response_active = False
    return None


def general_command_listener():
    global alert_response_active
    while True:
        if alert_response_active:
            time.sleep(1)
            continue
        logger.info("Listening for patient command...")
        with recognizer_lock:
            result = speech_recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            command = result.text.lower()
            logger.info(f"Patient said: {command}")
            responded = False
            for category, keywords in keyword_categories.items():
                for keyword in keywords:
                    if keyword in command:
                        response = responses.get(category, "I did not understand.")
                        speak(response)
                        responded = True
                        if category == "Help_Keywords":
                            send_alert("patient_needs_help")
                        break
                if responded:
                    break
            if not responded:
                speak("I did not understand. Can you repeat?")
        else:
            logger.info("No speech could be recognized.")
            time.sleep(1)


# ======= Utility Functions =======
def inside_rect(pt, rect_xyxy):
    (x1, y1, x2, y2) = rect_xyxy
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def is_inside_bed(pt, bed_box):
    if bed_box is None:
        return False
    x1, y1, x2, y2 = bed_box
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def point_side_of_line(pt, a, b):
    return np.sign((pt[0] - a[0]) * (b[1] - a[1]) - (pt[1] - a[1]) * (b[0] - a[0]))


# ======= Simple Object Tracker =======
class SimpleTracker:
    def __init__(self, max_distance=60, max_lost=8):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, centroids):
        obj_ids = list(self.objects.keys())
        unmatched_obj_ids = set(obj_ids)
        unmatched_centroids = list(range(len(centroids)))
        for oid in obj_ids:
            ocent = self.objects[oid]['centroid']
            best_idx = None
            best_dist = self.max_distance
            for i in unmatched_centroids:
                c = centroids[i]
                d = math.hypot(ocent[0] - c[0], ocent[1] - c[1])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None:
                self.objects[oid]['centroid'] = centroids[best_idx]
                self.objects[oid]['lost'] = 0
                unmatched_obj_ids.discard(oid)
                unmatched_centroids.remove(best_idx)
        to_delete = []
        for oid in unmatched_obj_ids:
            self.objects[oid]['lost'] += 1
            if self.objects[oid]['lost'] > self.max_lost:
                to_delete.append(oid)
        for oid in to_delete:
            del self.objects[oid]
        for i in unmatched_centroids:
            self.objects[self.next_id] = {
                'centroid': centroids[i],
                'lost': 0,
                'side': {1: None, 2: None, 3: None, 4: None},
                'last_alert': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                'inside_bed': None  # Added to track if object is inside bed
            }
            self.next_id += 1
        return self.objects


# ======= YOLOv8 Detector with Bed Detection =======
class YOLODetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.class_names = self.model.names

    def detect_bed_and_people(self, frame):
        results = self.model(frame, verbose=False)[0]
        bed_box = None
        person_rects = []

        for res in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = res
            if int(cls) == 0 and score > 0.3:  # person
                person_rects.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            elif self.class_names[int(cls)].lower() == 'bed' and score > 0.3:
                bed_box = (int(x1), int(y1), int(x2), int(y2))

        return bed_box, person_rects


# ======= Automatic Boundary Generator =======
class AutoBoundaryGenerator:
    def __init__(self, margin_percent=0.15):
        self.margin_percent = margin_percent

    def generate_boundaries(self, bed_box, frame_shape):
        """Generate four boundary lines around the detected bed"""
        if bed_box is None:
            return None

        bed_x1, bed_y1, bed_x2, bed_y2 = bed_box
        frame_height, frame_width = frame_shape[:2]

        # Calculate margin based on bed size
        margin_x = int((bed_x2 - bed_x1) * self.margin_percent)
        margin_y = int((bed_y2 - bed_y1) * self.margin_percent)

        # Create expanded boundary around bed
        boundary_x1 = max(0, bed_x1 - margin_x)
        boundary_y1 = max(0, bed_y1 - margin_y)
        boundary_x2 = min(frame_width, bed_x2 + margin_x)
        boundary_y2 = min(frame_height, bed_y2 + margin_y)

        # Define four boundary lines (left, right, top, bottom)
        boundaries = {
            'left': ((boundary_x1, boundary_y1), (boundary_x1, boundary_y2)),
            'right': ((boundary_x2, boundary_y1), (boundary_x2, boundary_y2)),
            'top': ((boundary_x1, boundary_y1), (boundary_x2, boundary_y1)),
            'bottom': ((boundary_x1, boundary_y2), (boundary_x2, boundary_y2))
        }

        return boundaries, (boundary_x1, boundary_y1, boundary_x2, boundary_y2)


# ======= Alert Interaction Thread =======
def alert_interaction(oid, boundary_side):
    global alert_response_active
    alert_response_active = True
    speak(f"Do you need any help getting up from the bed? Please say okay if you’re fine, or help if you need assistance.")
    reply = listen_for_keywords(
        keyword_categories,
        timeout=20,
        ok_response=responses["Ok_Keywords"],
        help_response=responses["Help_Keywords"],
        context="bed boundary alert"
    )
    if reply == "ok":
        logger.info(f"Patient acknowledged after {boundary_side} boundary crossing.")
    elif reply == "help":
        logger.info(f"Patient requested help after {boundary_side} boundary crossing.")
        send_alert("patient_needs_help")
    else:
        speak("No clear response received. Notifying your caretaker.")
        send_alert("patient_unresponsive")
    alert_response_active = False


# ======= Main Program =======
def main():
    global alert_response_active
    is_fullscreen = True
    cam_width, cam_height = 1280, 960

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (cam_width, cam_height)}))
    picam2.start()

    detector = YOLODetector()
    tracker = SimpleTracker(max_distance=70, max_lost=10)
    boundary_generator = AutoBoundaryGenerator(margin_percent=0.0)

    alert_cooldown = 1.5
    alert_messages = {1: "", 2: "", 3: "", 4: ""}
    alert_timestamps = {1: 0, 2: 0, 3: 0, 4: 0}
    alert_display_time = 2.0

    # Boundary sides mapping
    boundary_sides = {1: "left", 2: "right", 3: "top", 4: "bottom"}
    boundary_colors = {
        "left": (0, 0, 255),    # Red
        "right": (255, 0, 0),   # Blue
        "top": (0, 255, 255),   # Yellow
        "bottom": (255, 0, 255) # Magenta
    }

    current_boundaries = None
    current_boundary_box = None
    bed_detected = False

    # SET FULL FRAME ROI - This is the key change
    roi_xyxy = (0, 0, cam_width, cam_height)  # Full frame ROI

    threading.Thread(target=general_command_listener, daemon=True).start()

    cv2.namedWindow("Auto Bed Boundary Alert", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Auto Bed Boundary Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Detect bed and people
            bed_box, rects = detector.detect_bed_and_people(frame_rgb)

            # Generate automatic boundaries if bed is detected
            if bed_box is not None and not bed_detected:
                current_boundaries, current_boundary_box = boundary_generator.generate_boundaries(bed_box, frame_bgr.shape)
                if current_boundaries:
                    bed_detected = True
                    logger.info("Bed detected and boundaries set automatically")
                    speak("Bed detected. Monitoring system is now active.")

            # Process person detection and tracking - NOW WITH FULL FRAME ROI
            centroids = []
            filtered_rects = []

            for (x, y, w, h) in rects:
                cx, cy = x + w // 2, y + h // 2
                # Check if centroid is inside FULL FRAME ROI (which is the entire frame)
                if inside_rect((cx, cy), roi_xyxy):
                    centroids.append((cx, cy))
                    filtered_rects.append((x, y, w, h))

            # Pixelate detected person areas for privacy - ALL detected persons
            for (x, y, w, h) in filtered_rects:
                person_roi = frame_bgr[y:y+h, x:x+w]
                if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                    small = cv2.resize(person_roi, (30, 30), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    frame_bgr[y:y+h, x:x+w] = pixelated

            objects = tracker.update(centroids)

            # Draw FULL FRAME ROI (optional - for visualization)
            (rx1, ry1, rx2, ry2) = roi_xyxy
            cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)  # Green ROI for full frame

            # Draw bed and boundaries
            if bed_box is not None:
                bed_x1, bed_y1, bed_x2, bed_y2 = bed_box
                cv2.rectangle(frame_bgr, (bed_x1, bed_y1), (bed_x2, bed_y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, "Bed", (bed_x1, bed_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if current_boundaries is not None:
                # Draw boundary box
                bx1, by1, bx2, by2 = current_boundary_box
                cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

                # Draw boundary lines with different colors
                for idx, (side, line) in enumerate(current_boundaries.items()):
                    color = boundary_colors.get(side, (255, 255, 255))
                    cv2.line(frame_bgr, line[0], line[1], color, 3)
                    # Add side label
                    mid_x = (line[0][0] + line[1][0]) // 2
                    mid_y = (line[0][1] + line[1][1]) // 2
                    cv2.putText(frame_bgr, side, (mid_x - 20, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Track objects and check boundary crossings - ALL tracked objects in full frame
            for oid, info in objects.items():
                c = info['centroid']
                cv2.circle(frame_bgr, (int(c[0]), int(c[1])), 4, (0, 255, 255), -1)
                cv2.putText(frame_bgr, f'ID {oid}', (int(c[0]) + 6, int(c[1]) - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                # Update inside_bed status
                prev_inside = info.get('inside_bed', None)
                cur_inside = is_inside_bed(c, current_boundary_box) if current_boundary_box is not None else None
                info['inside_bed'] = cur_inside
                logger.debug(f"ID {oid}: Inside bed updated from {prev_inside} to {cur_inside}, centroid={c}")


                # Check boundary crossings only if boundaries and prev/cur inside states exist
                if current_boundaries is not None and prev_inside is not None and cur_inside is not None:
                    for idx, (side, line) in enumerate(current_boundaries.items()):
                        boundary_idx = list(current_boundaries.keys()).index(side) + 1
                        prev_side = info['side'].get(boundary_idx)
                        cur_side = int(point_side_of_line(c, line[0], line[1]))

                        if cur_side == 0:
                            continue

                        # Detect crossing event
                        if prev_side is not None and prev_side != 0 and prev_side != cur_side:
                            logger.debug(f"ID {oid}: Boundary crossed on {side} side. prev_side={prev_side}, cur_side={cur_side}, inside_bed prev={prev_inside}, cur={cur_inside}")
                            now = time.time()
                            # Only alert if patient is exiting bed (inside -> outside)
                            if prev_inside and not cur_inside:
                                if now - info['last_alert'].get(boundary_idx, 0) >= alert_cooldown and not alert_response_active:
                                    info['last_alert'][boundary_idx] = now
                                    alert_timestamps[boundary_idx] = now
                                    alert_messages[boundary_idx] = f"ALERT: {side.upper()} boundary crossed (ID {oid})"
                                    logger.info(alert_messages[boundary_idx])
                                    # Start alert interaction
                                    threading.Thread(target=alert_interaction, args=(oid, side), daemon=True).start()

                        info['side'][boundary_idx] = cur_side

            # Display alert messages
            current_time = time.time()
            y_offset = 30
            for idx in range(1, 5):
                if current_time - alert_timestamps[idx] < alert_display_time:
                    cv2.putText(frame_bgr, alert_messages[idx], (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25

            # Display status
            status_text = "Bed detected: YES" if bed_detected else "Searching for bed..."
            cv2.putText(frame_bgr, status_text, (10, frame_bgr.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(frame_bgr, "q:quit  m:toggle fullscreen  r:reset bed detection",
                       (10, frame_bgr.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            cv2.imshow("Auto Bed Boundary Alert", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                if is_fullscreen:
                    cv2.setWindowProperty("Auto Bed Boundary Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Auto Bed Boundary Alert", 800, 600)
                    is_fullscreen = False
                else:
                    cv2.setWindowProperty("Auto Bed Boundary Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True
            elif key == ord('r'):
                # Reset bed detection
                bed_detected = False
                current_boundaries = None
                current_boundary_box = None
                logger.info("Bed detection reset - searching for bed again")

            # FPS calculation and print
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:  # Every second
                logger.info(f"Processed {frame_count} frames per second")
                frame_count = 0
                prev_time = now

    finally:
        picam2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
