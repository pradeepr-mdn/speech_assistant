
#2(chair and bed)  
import os
import time
import math
import threading
import uuid
import csv
import logging
from datetime import datetime
import cv2
import numpy as np
import pytz
from picamera2 import Picamera2
from ultralytics import YOLO
import azure.cognitiveservices.speech as speechsdk
import requests
from dotenv import load_dotenv

load_dotenv()

# ======= Logging =======
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("picamera2").setLevel(logging.INFO)

# ======= Azure Speech SDK Configuration =======
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
if not speech_key or not service_region:
    logger.error("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION missing in environment")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "en-IN"

speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
speech_config.set_property(
    speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "8000"
)

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

tts_lock = threading.Lock()
rec_lock = threading.Lock()

SERVER_ALERT_URL = "https://sens-able.manipal.net/api/alerts/alerts/"

responses = {
    "Help_Keywords": "I'll notify your caretaker, help is on the way.",
    "Hello_Keywords": "Hello! How can I assist you today?",
    "Ok_Keywords": "Glad to hear you're alright.",
}

alert_response_active = False
alert_handling_active = threading.Event()
initial_bed_speech_done = False


def get_mac_address():
    mac_int = uuid.getnode()
    mac = ':'.join(['{:02x}'.format((mac_int >> ele) & 0xff) for ele in range(40, -1, -8)])
    return mac

def get_ist_time():
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
            resp = requests.post(SERVER_ALERT_URL, json=alert_data, timeout=8)
            if resp.status_code == 200:
                logger.info(f"Alert sent: {alert_content}")
            else:
                logger.warning(f"Failed to send alert: {resp.status_code}")
        except Exception as e:
            logger.exception(f"Error sending alert: {e}")
    threading.Thread(target=_send, daemon=True).start()

def load_keywords_from_csv(csv_filepath):
    categories = {}
    try:
        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for category, keywords_str in row.items():
                    if category not in categories:
                        categories[category] = set()
                    if keywords_str:
                        keywords = {k.strip().lower() for k in keywords_str.split(',') if k.strip()}
                        categories[category].update(keywords)
    except Exception as e:
        logger.exception(f"Failed to load keywords CSV: {e}")
    return categories

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "data/HelpKeywords-HelloKeywords-OkKeywords.csv")
keyword_categories = load_keywords_from_csv(csv_path)

def speak(text):
    with tts_lock:
        try:
            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Spoken: {text}")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cd = result.cancellation_details
                logger.error(f"TTS canceled: {cd.reason}; {cd.error_details}")
            else:
                logger.warning(f"TTS unexpected result: {result.reason}")
        except:
            logger.exception("TTS failed")
        time.sleep(0.12)

def make_recognizer():
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    r = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    return r

def listen_for_keywords(keyword_categories, timeout=20, ok_response=None, help_response=None, context="interaction"):
    global alert_response_active
    start_time = time.time()
    local_recognizer = make_recognizer()

    while time.time() - start_time < timeout:
        logger.debug(f"[{context}] Listening... ({int(timeout - (time.time()-start_time))}s left)")
        try:
            result = local_recognizer.recognize_once_async().get()
        except:
            logger.exception("Recognition failed")
            time.sleep(0.5)
            continue

        if result is None:
            time.sleep(0.3)
            continue

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            command = (result.text or "").lower()
            logger.info(f"[{context}] Recognized: {command}")

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

            time.sleep(0.3)
        elif result.reason == speechsdk.ResultReason.NoMatch:
            time.sleep(0.3)
        elif result.reason == speechsdk.ResultReason.Canceled:
            time.sleep(1)
        else:
            time.sleep(0.3)
    alert_response_active = False
    return None

class ContinuousGeneralListener(threading.Thread):
    def __init__(self, keyword_categories, responses):
        super().__init__(daemon=True)
        self.keyword_categories = keyword_categories
        self.responses = responses
        self._stop_event = threading.Event()
    def run(self):
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True))
        def recognized(evt):
            try:
                result = evt.result
                if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                    return
                command = (result.text or "").lower()
                if alert_response_active or len(command.strip()) < 2:
                    return
                logger.info(f"[{command}]")
                responded = False
                for category, keywords in self.keyword_categories.items():
                    for keyword in keywords:
                        if keyword in command:
                            response = self.responses.get(category, "I did not understand.")
                            threading.Thread(target=speak, args=(response,), daemon=True).start()
                            responded = True
                            if category == "Help_Keywords":
                                send_alert("patient_needs_help")
                            break
                    if responded:
                        break
            except:
                logger.exception("recognize exception")
        def canceled(evt):
            try:
                if hasattr(evt, 'result') and evt.result is not None:
                    cd = evt.result.cancellation_details
                    logger.warning(f"[continuous] Canceled: {cd.reason}; {cd.error_details}")
            except:
                logger.exception("canceled exception")
        def session_stopped(evt):
            logger.info("[continuous] Session stopped")
            self._stop_event.set()

        recognizer.recognized.connect(recognized)
        recognizer.canceled.connect(canceled)
        recognizer.session_stopped.connect(session_stopped)

        logger.info("[continuous] start recognition")
        try:
            recognizer.start_continuous_recognition()
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except:
            logger.exception("continuous exception")
        finally:
            try:
                recognizer.stop_continuous_recognition()
            except:
                pass
    def stop(self):
        self._stop_event.set()

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
                'inside_bed': None,
                'inside_chair': None,
                'last_chair_alert': 0.0, # NEW: Track last chair alert time
                'prev_upper_y': 0.0,
                'rising_frames': 0, # NEW: Count consecutive rising frames
                'total_movement': 0.0  # NEW: Track cumulative upward movement
            }
            self.next_id += 1
        return self.objects


class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.class_names = self.model.names

    def detect_bed_chair_and_people(self, frame):
        results = self.model(frame, verbose=False)[0]
        bed_box, chair_box = None, None
        person_rects = []

        for res in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = res
            cls_name = self.class_names[int(cls)].lower()
            if cls_name == 'person' and score > 0.3:
                person_rects.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            elif cls_name == 'bed' and score > 0.3:
                bed_box = (int(x1), int(y1), int(x2), int(y2))
            elif cls_name == 'chair' and score > 0.3:
                chair_box = (int(x1), int(y1), int(x2), int(y2))

        return bed_box, chair_box, person_rects


class AutoBoundaryGenerator:
    def __init__(self, margin_percent=0.15, top_offset_pixels=100):
        self.margin_percent = margin_percent
        self.top_offset_pixels = top_offset_pixels

    def generate_boundaries(self, bed_box, frame_shape):
        if bed_box is None:
            return None
        bed_x1, bed_y1, bed_x2, bed_y2 = bed_box
        frame_height, frame_width = frame_shape[:2]
        margin_x = int((bed_x2 - bed_x1) * self.margin_percent)
        margin_y = int((bed_y2 - bed_y1) * self.margin_percent)
        boundary_x1 = max(0, bed_x1 - margin_x)
        boundary_y1 = max(0, bed_y1 - margin_y - self.top_offset_pixels)
        boundary_x2 = min(frame_width, bed_x2 + margin_x)
        boundary_y2 = min(frame_height, bed_y2 + margin_y)

        boundaries = {
            'left': ((boundary_x1, boundary_y1), (boundary_x1, boundary_y2)),
            'right': ((boundary_x2, boundary_y1), (boundary_x2, boundary_y2)),
            'top': ((boundary_x1, boundary_y1), (boundary_x2, boundary_y1)),
            'bottom': ((boundary_x1, boundary_y2), (boundary_x2, boundary_y2))
        }
        return boundaries, (boundary_x1, boundary_y1, boundary_x2, boundary_y2)

def inside_rect(pt, rect_xyxy):
    (x1, y1, x2, y2) = rect_xyxy
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def get_bbox_edges(x, y, w, h):
    return [
        ((x, y), (x + w, y)),
        ((x, y + h), (x + w, y + h)),
        ((x, y), (x, y + h)),
        ((x + w, y), (x + w, y + h))
    ]

def line_segments_intersect(p1, p2, q1, q2):
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    def on_segment(a, b, c):
        return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
                min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True
    return False

def alert_interaction_bed(oid, boundary_side):
    """Alert interaction for bed boundary crossing"""
    global alert_response_active
    alert_response_active = True
    alert_handling_active.set()

    speak("Do you need any help getting up from the bed? Please say okay if you're fine, or help if you need assistance.")

    reply = listen_for_keywords(
        keyword_categories,
        timeout=20,
        ok_response=responses["Ok_Keywords"],
        help_response=responses["Help_Keywords"],
        context="bed boundary alert"
    )

    if reply == "ok":
        logger.info(f"Patient acknowledged after {boundary_side} bed boundary crossing.")
    elif reply == "help":
        logger.info(f"Patient requested help after {boundary_side} bed boundary crossing.")
        send_alert("patient_needs_help_bed")
    else:
        speak("No clear response received. Notifying your caretaker.")
        send_alert("patient_unresponsive_bed")

    alert_response_active = False
    alert_handling_active.clear()


def alert_interaction_chair(oid, boundary_side):
    """Alert interaction for chair pre-exit detection"""
    global alert_response_active
    alert_response_active = True
    alert_handling_active.set()

    speak("Do you need any help getting up from the chair? Please say okay if you're fine, or help if you need assistance.")

    reply = listen_for_keywords(
        keyword_categories,
        timeout=20,
        ok_response=responses["Ok_Keywords"],
        help_response=responses["Help_Keywords"],
        context="chair pre-exit alert"
    )

    if reply == "ok":
        logger.info(f"Patient acknowledged chair pre-exit alert.")
    elif reply == "help":
        logger.info(f"Patient requested help for chair.")
        send_alert("patient_needs_help_chair")
    else:
        speak("No clear response received. Notifying your caretaker.")
        send_alert("patient_unresponsive_chair")

    alert_response_active = False
    alert_handling_active.clear()

    
def bed_boundary_reset_worker():
    global bed_detected, current_boundaries, current_boundary_box
    while True:
        if bed_detected:
            time.sleep(18000)  # Wait for 5 hour
            logger.info("Automatic 5 hour bed boundary reset")
            # Reset boundaries silently in background; no speech
            bed_detected = False
            current_boundaries = None
            current_boundary_box = None
            global initial_bed_speech_done
        else:
            time.sleep(10)  # Check every 10s if bed is detected


def main():
    global alert_response_active, bed_detected, current_bed_boundaries, bed_boundary_box
    global chair_detected, current_chair_boundaries, chair_boundary_box

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

    boundary_colors = {
        "left": (0, 0, 255),
        "right": (255, 0, 0),
        "top": (0, 255, 255),
        "bottom": (255, 0, 255)
    }

    current_bed_boundaries = None
    bed_boundary_box = None
    bed_detected = False

    current_chair_boundaries = None
    chair_boundary_box = None
    chair_detected = False

    roi_xyxy = (0, 0, cam_width, cam_height)

    continuous_listener = ContinuousGeneralListener(keyword_categories, responses)
    continuous_listener.start()

    threading.Thread(target=bed_boundary_reset_worker, daemon=True).start()

    cv2.namedWindow("Auto Bed & Chair Monitor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Auto Bed & Chair Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = time.time()
    frame_count = 0
    target_fps = 10
    frame_interval = 1.0 / target_fps
    last_frame_time = 0

    # CHAIR PRE-EXIT MOVEMENT THRESHOLD
    RISE_THRESHOLD = 15  # or 20
    
    # Periodic chair boundary update interval (5 minutes)
    CHAIR_UPDATE_INTERVAL = 300
    last_chair_update_time = 0

    try:
        while True:
            now = time.time()
            if now - last_frame_time < frame_interval:
                time.sleep(frame_interval - (now - last_frame_time))
                continue
            last_frame_time = now

            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            bed_box, chair_box, rects = detector.detect_bed_chair_and_people(frame_rgb)

            # Bed detection
            if bed_box is not None and not bed_detected:
                current_bed_boundaries, bed_boundary_box = boundary_generator.generate_boundaries(bed_box, frame_bgr.shape)
                if current_bed_boundaries:
                    bed_detected = True
                    global initial_bed_speech_done
                    if not initial_bed_speech_done:
                        speak("Patient monitoring system is active.")
                        initial_bed_speech_done = True

            # Chair detection with periodic boundary update
            if chair_box is not None:
                if (not chair_detected) or (now - last_chair_update_time) > CHAIR_UPDATE_INTERVAL:
                    current_chair_boundaries, chair_boundary_box = boundary_generator.generate_boundaries(chair_box, frame_bgr.shape)
                    chair_detected = True
                    last_chair_update_time = now
                    logger.info("Chair boundary updated (periodic)")
            else:
                # Chair not detected in this frame, optionally keep previous boundaries or reset flags if required
                pass

            centroids = []
            filtered_rects = []

            for (x, y, w, h) in rects:
                cx, cy = x + w // 2, y + h // 2
                if inside_rect((cx, cy), roi_xyxy):
                    centroids.append((cx, cy))
                    filtered_rects.append((x, y, w, h))

            # Draw person boxes and pixelate
            for (x, y, w, h) in filtered_rects:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
                person_roi = frame_bgr[y:y+h, x:x+w]
                if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                    small = cv2.resize(person_roi, (30, 30), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    frame_bgr[y:y+h, x:x+w] = pixelated
                upper_body_y = int((y + h) - (h * 0.7))
                torso_center_x = int(x + w // 2)
                cv2.circle(frame_bgr, (torso_center_x, upper_body_y), 6, (0, 0, 255), -1)  # Large red dot
                cv2.putText(
                    frame_bgr,
                    "Torso",
                    (torso_center_x + 8, upper_body_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )                
            objects = tracker.update(centroids)

            for i, oid in enumerate(objects.keys()):
                if i < len(filtered_rects):
                    objects[oid]['bbox'] = filtered_rects[i]

            # Draw ROI
            (rx1, ry1, rx2, ry2) = roi_xyxy
            cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

            # Draw bed
            if bed_detected and bed_boundary_box:
                bx1, by1, bx2, by2 = bed_boundary_box
                cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, "Bed", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                for side, line in current_bed_boundaries.items():
                    color = boundary_colors.get(side, (255, 255, 255))
                    cv2.line(frame_bgr, line[0], line[1], color, 3)
                    mid_x = (line[0][0] + line[1][0]) // 2
                    mid_y = (line[0][1] + line[1][1]) // 2
                    cv2.putText(frame_bgr, side, (mid_x - 20, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw chair
            if chair_detected and chair_boundary_box:
                cx1, cy1, cx2, cy2 = chair_boundary_box
                
                # Outer chair boundary (orange)
                cv2.rectangle(frame_bgr, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)
                cv2.putText(frame_bgr, "Chair", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                # Chair boundary lines
                for side, line in current_chair_boundaries.items():
                    color = (255, 180, 0)
                    cv2.line(frame_bgr, line[0], line[1], color, 3)

                # Inner safe zone (cyan) - for debugging
                inner_margin = 10
                safe_cx1 = cx1 + inner_margin
                safe_cy1 = cy1 + inner_margin
                safe_cx2 = cx2 - inner_margin
                safe_cy2 = cy2 - inner_margin
                cv2.rectangle(frame_bgr, (safe_cx1, safe_cy1), (safe_cx2, safe_cy2), (0, 255, 255), 2)
                cv2.putText(frame_bgr, "Safe Zone", (safe_cx1 + 5, safe_cy1 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # ===== BED BOUNDARY CROSSING DETECTION =====
            if not alert_handling_active.is_set() and current_bed_boundaries is not None:
                for oid, info in objects.items():
                    bbox = info.get('bbox')
                    if bbox is None:
                        continue
                    x, y, w, h = bbox
                    person_edges = get_bbox_edges(x, y, w, h)
                    centroid = info.get('centroid')

                    prev_inside = info.get('inside_bed', None)
                    cur_inside = None
                    if centroid is not None and bed_boundary_box is not None:
                        cur_inside = inside_rect(centroid, bed_boundary_box)
                    info['inside_bed'] = cur_inside

                    logger.debug(f"Bed - Object {oid}: prev_inside={prev_inside}, cur_inside={cur_inside}")

                    # Check boundary line crossing
                    for idx, (side, line) in enumerate(current_bed_boundaries.items()):
                        boundary_idx = idx + 1
                        for edge in person_edges:
                            if line_segments_intersect(edge[0], edge[1], line[0], line[1]):
                                if prev_inside and cur_inside:
                                    now = time.time()
                                    if now - info['last_alert'].get(boundary_idx, 0) >= alert_cooldown and not alert_response_active:
                                        info['last_alert'][boundary_idx] = now
                                        alert_timestamps[boundary_idx] = now
                                        alert_messages[boundary_idx] = f"ALERT: {side.upper()} bed boundary crossed (ID {oid})"
                                        logger.info(alert_messages[boundary_idx])
                                        threading.Thread(target=alert_interaction_bed, args=(oid, side), daemon=True).start()
                                break

            # ===== CHAIR PRE-EXIT DETECTION (SINGLE FRAME, HIGHER THRESHOLD) =====
            if chair_detected and chair_boundary_box is not None and not alert_handling_active.is_set():
                for oid, info in objects.items():
                    bbox = info.get('bbox')
                    c = info.get('centroid')
                    if c is None or bbox is None:
                        continue

                    x, y, w, h = bbox

                    # Check if centroid inside chair safe zone
                    prev_inside_chair = info.get('inside_chair', False)
                    inside_chair = inside_rect(c, (safe_cx1, safe_cy1, safe_cx2, safe_cy2))
                    info['inside_chair'] = inside_chair

                    now = time.time()
                    seated_time = info.get('seated_time', None)

                    # Detect entry: outside to inside
                    if not prev_inside_chair and inside_chair:
                        info['seated_time'] = now
                        logger.info(f"Object {oid} entered chair.")

                    # Detect exit: inside to outside
                    if prev_inside_chair and not inside_chair:
                        info['seated_time'] = None
                        logger.info(f"Object {oid} exited chair.")

                    # Start rising detection only if inside chair and seated delay passed
                    if inside_chair and seated_time and (now - seated_time > 5.0):
                        upper_body_y = (y + h) - (h * 0.4)
                        prev_upper_y = info.get('prev_upper_y', upper_body_y)
                        vertical_movement = prev_upper_y - upper_body_y
                        is_rising_now = vertical_movement > RISE_THRESHOLD

                        logger.debug(
                            f"[Chair-Rise Debug] ID:{oid} | upper_y:{upper_body_y:.2f} "
                            f"prev_y:{prev_upper_y:.2f} | move:{vertical_movement:.2f} | "
                            f"rising:{is_rising_now}"
                        )

                        if is_rising_now:
                            last_chair_alert = info.get('last_chair_alert', 0)
                            time_since_last = now - last_chair_alert

                            if time_since_last >= 10:
                                info['last_chair_alert'] = now
                                logger.warning(f"🚨 CHAIR PRE-EXIT ALERT TRIGGERED for ID {oid} 🚨")
                                threading.Thread(target=alert_interaction_chair, args=(oid, 'chair'), daemon=True).start()

                        info['prev_upper_y'] = upper_body_y

            # Overlay alerts
            current_time = time.time()
            y_offset = 30
            for idx in range(1, 5):
                if current_time - alert_timestamps.get(idx, 0) < alert_display_time:
                    cv2.putText(frame_bgr, alert_messages[idx], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25

            # Status text
            status_bed = "Bed: YES" if bed_detected else "Bed: NO"
            status_chair = "Chair: YES" if chair_detected else "Chair: NO"
            status_text = f"{status_bed} | {status_chair}"
            cv2.putText(frame_bgr, status_text, (10, frame_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Auto Bed & Chair Monitor", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                if is_fullscreen:
                    cv2.setWindowProperty("Auto Bed & Chair Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Auto Bed & Chair Monitor", 800, 600)
                    is_fullscreen = False
                else:
                    cv2.setWindowProperty("Auto Bed & Chair Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True

            frame_count += 1
            if time.time() - prev_time >= 1.0:
                logger.info(f"FPS: {frame_count}")
                frame_count = 0
                prev_time = time.time()

    finally:
        try:
            continuous_listener.stop()
        except:
            pass
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
