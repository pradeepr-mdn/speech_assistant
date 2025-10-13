import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
from picamera2 import Picamera2

def inside_rect(pt, rect_xyxy):
    x1, y1, x2, y2 = rect_xyxy
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
        for oid in unmatched_obj_ids:
            self.objects[oid]['lost'] += 1
        to_delete = [oid for oid in unmatched_obj_ids if self.objects[oid]['lost'] > self.max_lost]
        for oid in to_delete:
            del self.objects[oid]
        for i in unmatched_centroids:
            self.objects[self.next_id] = {'centroid': centroids[i], 'lost': 0, 'side': {}, 'last_alert': {}}
            self.next_id += 1
        return self.objects

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
            cls = int(cls)
            if cls == 0 and score > 0.3:  # person class idx 0
                person_rects.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            elif self.class_names[cls].lower() == 'bed' and score > 0.3:
                bed_box = (int(x1), int(y1), int(x2), int(y2))
        return bed_box, person_rects

class AutoBoundaryGenerator:
    def __init__(self, margin_percent=0.15):
        self.margin_percent = margin_percent
    
    def generate_boundaries(self, bed_box, frame_shape):
        if bed_box is None:
            return None
        bed_x1, bed_y1, bed_x2, bed_y2 = bed_box
        frame_height, frame_width = frame_shape[:2]
        margin_x = int((bed_x2 - bed_x1) * self.margin_percent)
        margin_y = int((bed_y2 - bed_y1) * self.margin_percent)
        boundary_x1 = max(0, bed_x1 - margin_x)
        boundary_y1 = max(0, bed_y1 - margin_y)
        boundary_x2 = min(frame_width, bed_x2 + margin_x)
        boundary_y2 = min(frame_height, bed_y2 + margin_y)
        boundaries = {
            'left': ((boundary_x1, boundary_y1), (boundary_x1, boundary_y2)),
            'right': ((boundary_x2, boundary_y1), (boundary_x2, boundary_y2)),
            'top': ((boundary_x1, boundary_y1), (boundary_x2, boundary_y1)),
            'bottom': ((boundary_x1, boundary_y2), (boundary_x2, boundary_y2))
        }
        return boundaries, (boundary_x1, boundary_y1, boundary_x2, boundary_y2)

def main():
    cam_width, cam_height = 1280, 960

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (cam_width, cam_height)})
    picam2.configure(preview_config)
    picam2.start()

    detector = YOLODetector()
    tracker = SimpleTracker(max_distance=70, max_lost=10)
    boundary_generator = AutoBoundaryGenerator(margin_percent=0.15)

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

    current_boundaries = None
    current_boundary_box = None
    bed_detected = False
    roi_xyxy = (0, 0, cam_width, cam_height)

    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bed_box, rects = detector.detect_bed_and_people(frame_rgb)

        if bed_box is not None and not bed_detected:
            current_boundaries, current_boundary_box = boundary_generator.generate_boundaries(bed_box, frame.shape)
            bed_detected = True
            print("Bed detected. Boundaries set.")

        centroids = []
        filtered_rects = []
        for (x, y, w, h) in rects:
            cx, cy = x + w // 2, y + h // 2
            if inside_rect((cx, cy), roi_xyxy):
                centroids.append((cx, cy))
                filtered_rects.append((x, y, w, h))

        for (x, y, w, h) in filtered_rects:
            person_roi = frame[y:y+h, x:x+w]
            if person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                small = cv2.resize(person_roi, (30, 30), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = pixelated

        objects = tracker.update(centroids)

        rx1, ry1, rx2, ry2 = roi_xyxy
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        if bed_box is not None:
            bed_x1, bed_y1, bed_x2, bed_y2 = bed_box
            cv2.rectangle(frame, (bed_x1, bed_y1), (bed_x2, bed_y2), (0, 255, 0), 2)
            cv2.putText(frame, "Bed", (bed_x1, bed_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if current_boundaries is not None:
            bx1, by1, bx2, by2 = current_boundary_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
            for idx, (side, line) in enumerate(current_boundaries.items()):
                color = boundary_colors.get(side, (255, 255, 255))
                cv2.line(frame, line[0], line[1], color, 3)
                mid_x = (line[0][0] + line[1][0]) // 2
                mid_y = (line[0][1] + line[1][1]) // 2
                cv2.putText(frame, side, (mid_x - 20, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        now = time.time()
        for oid, info in objects.items():
            c = info['centroid']
            cv2.circle(frame, (int(c[0]), int(c[1])), 4, (0, 255, 255), -1)
            cv2.putText(frame, f'ID {oid}', (int(c[0]) + 6, int(c[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            if current_boundaries is not None:
                for idx, (side, line) in enumerate(current_boundaries.items()):
                    boundary_idx = idx + 1
                    prev_side = info['side'].get(boundary_idx)
                    cur_side = int(point_side_of_line(c, line[0], line[1]))

                    if cur_side == 0:
                        continue

                    if prev_side is not None and prev_side != 0 and prev_side != cur_side:
                        if now - info['last_alert'].get(boundary_idx, 0) >= alert_cooldown:
                            info['last_alert'][boundary_idx] = now
                            alert_timestamps[boundary_idx] = now
                            alert_messages[boundary_idx] = f"ALERT: {side.upper()} boundary crossed (ID {oid})"
                            print(alert_messages[boundary_idx])  # Print alert instead of sending

                    info['side'][boundary_idx] = cur_side

        y_offset = 30
        for idx in range(1, 5):
            if now - alert_timestamps[idx] < alert_display_time:
                cv2.putText(frame, alert_messages[idx], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 25

        cv2.imshow("YOLOv8 Bed & Person Detection Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
