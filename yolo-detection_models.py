import cv2
import numpy as np
import time
import math
import threading
import os
import csv
import tempfile
import wave
import sounddevice as sd
from datetime import datetime
from ultralytics import YOLO
from picamera2 import Picamera2
import requests
import uuid
import pytz
import logging
import onnx_asr
import kokoro_tts

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('onnxruntime').setLevel(logging.WARNING)

# ======= Setup console logging (no file) =======
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# ======= Speech & TTS Setup =======
recognizer_lock = threading.Lock()
stt_model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
kokoro_model_path = "/home/mdn/Desktop/work/speech-assistant/models/kokoro-v1.0.onnx"
kokoro_voices_path = "/home/mdn/Desktop/work/speech-assistant/models/voices-v1.0.bin"

print("Current working dir:", os.getcwd())

def record_audio(duration=3, fs=16000):
    print("Start speaking...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Done recording")
    return audio.flatten()

def save_temp_wav(audio, fs=16000):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f.name, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(fs)
            wav_file.writeframes(audio.tobytes())
        return f.name

def recognize_speech(timeout=3):
    with recognizer_lock:
        audio = record_audio(duration=timeout)
        wav_path = save_temp_wav(audio)
        recognized_text = stt_model.recognize(wav_path)
        logger.info(f"Recognized speech: {recognized_text}")
        return recognized_text.lower()

def speak(text):
    with recognizer_lock:
        with tempfile.NamedTemporaryFile("w", suffix='.txt', delete=False) as f:
            f.write(text)
            temp_text_path = f.name
        output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        kokoro_tts.convert_text_to_audio(
            input_file=temp_text_path,
            output_file=output_wav,
            voice="af_bella",
            speed=1.0,
            lang="en-us",
            format="wav",
            model_path=kokoro_model_path,
            voices_path=kokoro_voices_path
        )
        import soundfile as sf
        data, fs = sf.read(output_wav, dtype='float32')
        sd.play(data, fs)
        sd.wait()

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

fall_response_active = False
patient_id = 1  # Set this to actual patient ID

SERVER_ALERT_URL = "http://192.168.11.194:8000/alerts/alerts"

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
            resp = requests.post(SERVER_ALERT_URL, json=alert_data)
            if resp.status_code == 200:
                logger.info(f"Alert sent successfully: {alert_content}")
            else:
                logger.info(f"Failed to send alert: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.info(f"Error sending alert: {e}")
    threading.Thread(target=_send, daemon=True).start()

def listen_for_keywords(keyword_categories, timeout=60, ok_response=None, help_response=None, context="activity response"):
    global fall_response_active
    start_time = time.time()
    while time.time() - start_time < timeout:
        logger.info(f"Listening for {context}...")
        command = recognize_speech(timeout=3)
        if command:
            logger.info(f"Patient said: {command}")
            for keyword in keyword_categories.get("Ok_Keywords", []):
                if keyword in command:
                    if ok_response:
                        speak(ok_response)
                    fall_response_active = False
                    return "ok"
            for keyword in keyword_categories.get("Help_Keywords", []):
                if keyword in command:
                    if help_response:
                        speak(help_response)
                    send_alert("patient_needs_help")
                    fall_response_active = False
                    return "help"
        else:
            logger.info("No or unrecognized response, listening again...")
            time.sleep(2)
    fall_response_active = False
    return None

def general_command_listener():
    global fall_response_active
    while True:
        if fall_response_active:
            time.sleep(1)
            continue
        logger.info("Listening for patient command...")
        command = recognize_speech(timeout=3)
        if command:
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

def inside_rect(pt, rect_xyxy):
    (x1, y1, x2, y2) = rect_xyxy
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2

def point_side_of_line(pt, a, b):
    return np.sign((pt[0] - a[0]) * (b[1] - a[1]) - (pt[1] - a[1]) * (b[0] - a[0]))

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
                'side': {1: None, 2: None},
                'last_alert': {1: 0.0, 2: 0.0}
            }
            self.next_id += 1
        return self.objects

class YoloPeopleDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Small nano model for speed

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        rects = []
        for res in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = res
            if int(cls) == 0 and score > 0.3:  # Class 0 = person
                rects.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return rects

def alert_interaction(oid, idx):
    global fall_response_active
    fall_response_active = True
    speak("Are you okay? Do you need any help? Please say okay if you are fine, or help if you need assistance.")
    reply = listen_for_keywords(
        keyword_categories,
        timeout=20,
        ok_response=responses["Ok_Keywords"],
        help_response=responses["Help_Keywords"],
        context="line crossing alert"
    )
    if reply == "ok":
        logger.info(f"Patient acknowledged after line {idx} crossing.")
    elif reply == "help":
        logger.info(f"Patient requested help after line {idx} crossing.")
        send_alert("patient_needs_help")
    else:
        speak("No clear response received. Notifying your caretaker.")
        send_alert("patient_unresponsive")
    fall_response_active = False

def main():
    global fall_response_active
    is_fullscreen = True
    cam_width, cam_height = 640, 480
    roi_xyxy = (80, 60, 560, 420)
    line1 = ((200, 60), (120, 420))  # left line
    line2 = ((440, 60), (520, 420))  # right line
    alert_cooldown = 1.5
    alert_messages = {1: "", 2: ""}
    alert_timestamps = {1: 0, 2: 0}
    alert_display_time = 2.0  # seconds

    # ==== Picamera2 Setup ====
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (cam_width, cam_height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    detector = YoloPeopleDetector()
    tracker = SimpleTracker(max_distance=70, max_lost=10)
    selecting_line = 0
    tmp_points = []

    threading.Thread(target=general_command_listener, daemon=True).start()

    def on_mouse(event, x, y, flags, param):
        nonlocal selecting_line, tmp_points, line1, line2
        if event == cv2.EVENT_LBUTTONDOWN and selecting_line in (1, 2):
            tmp_points.append((x, y))
            if len(tmp_points) == 2:
                if selecting_line == 1:
                    line1 = (tmp_points[0], tmp_points[1])
                else:
                    line2 = (tmp_points[0], tmp_points[1])
                selecting_line = 0
                tmp_points.clear()

    cv2.namedWindow("ROI Line Alert", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ROI Line Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("ROI Line Alert", on_mouse)

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rects = detector.detect(frame)
            centroids = []
            filtered_rects = []
            for (x, y, w, h) in rects:
                cx, cy = x + w // 2, y + h // 2
                if inside_rect((cx, cy), roi_xyxy):
                    centroids.append((cx, cy))
                    filtered_rects.append((x, y, w, h))

            # Pixelate detected person areas for privacy
            for (x, y, w, h) in filtered_rects:
                person_roi = frame_bgr[y:y+h, x:x+w]
                small = cv2.resize(person_roi, (30, 30), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame_bgr[y:y+h, x:x+w] = pixelated

            objects = tracker.update(centroids)

            (rx1, ry1, rx2, ry2) = roi_xyxy
            cv2.rectangle(frame_bgr, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)  # Green ROI
            cv2.line(frame_bgr, line1[0], line1[1], (0, 0, 255), 2)   # Red line
            cv2.line(frame_bgr, line2[0], line2[1], (255, 0, 0), 2)   # Blue line

            for p in tmp_points:
                cv2.circle(frame_bgr, p, 5, (0, 255, 255), -1)  # Yellow points

            for oid, info in objects.items():
                c = info['centroid']
                cv2.circle(frame_bgr, (int(c[0]), int(c[1])), 4, (0, 255, 255), -1)  # Yellow centroid
                cv2.putText(frame_bgr, f'ID {oid}', (int(c[0]) + 6, int(c[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                for idx, line in ((1, line1), (2, line2)):
                    prev_side = info['side'][idx]
                    cur_side = int(point_side_of_line(c, line[0], line[1]))
                    if cur_side == 0:
                        continue
                    if prev_side is not None and prev_side != 0 and prev_side != cur_side:
                        now = time.time()
                        if now - info['last_alert'][idx] >= alert_cooldown and not fall_response_active:
                            info['last_alert'][idx] = now
                            alert_timestamps[idx] = now
                            alert_messages[idx] = f"ALERT: Line {idx} crossed (ID {oid})"
                            logger.info(alert_messages[idx])
                            threading.Thread(target=alert_interaction, args=(oid, idx), daemon=True).start()
                    info['side'][idx] = cur_side

            current_time = time.time()
            if current_time - alert_timestamps[1] < alert_display_time:
                cv2.putText(frame_bgr, alert_messages[1], (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if current_time - alert_timestamps[2] < alert_display_time:
                cv2.putText(frame_bgr, alert_messages[2], (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame_bgr, "q:quit  s:select ROI  1:set Line1  2:set Line2",
                        (10, frame_bgr.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            cv2.imshow("ROI Line Alert", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sel = cv2.selectROI("ROI Line Alert", frame_bgr, showCrosshair=True, fromCenter=False)
                try:
                    cv2.destroyWindow("ROI selector")
                except Exception:
                    pass
                x, y, w, h = sel
                if w > 0 and h > 0:
                    roi_xyxy = (int(x), int(y), int(x + w), int(y + h))
            elif key == ord('1'):
                selecting_line = 1
                tmp_points.clear()
            elif key == ord('2'):
                selecting_line = 2
                tmp_points.clear()
            elif key == ord('m'):
                if is_fullscreen:
                    cv2.setWindowProperty("ROI Line Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("ROI Line Alert", 800, 600)
                    is_fullscreen = False
                else:
                    cv2.setWindowProperty("ROI Line Alert", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    is_fullscreen = True

    finally:
        picam2.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
