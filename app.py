import os
import time
import math
import random
import threading
from collections import defaultdict

import cv2
import cvzone
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

class TrackerApp:
    def __init__(self):
        self.screenshots_dir = "screenshots"
        self.videos_dir = "videos"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        print("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        self.classNames = self.model.names

        self.is_recording = False
        self.out = None
        self.current_frame = None
        self.latest_frame_bytes = None
        self.frame_condition = threading.Condition()
        self.lock = threading.Lock()

        self.track_paths = defaultdict(list)
        self.prev_frame_time = 0

        # Stats
        self.current_fps = 0
        self.objects_count = 0

    def generate_fallback_image(self, message):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    def process_video(self):
        stream_url_env = os.environ.get("STREAM_URL", "0")
        stream_url = int(stream_url_env) if stream_url_env.isdigit() else stream_url_env

        print(f"Connecting to Camera at: {stream_url}")
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            print("Camera failed to connect over IP! Falling back to built-in webcam (0)...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Webcam also failed! Using fallback image stream.")
                cap = None

        if cap:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0: fps = 30
        else:
            frame_width, frame_height, fps = 640, 480, 30

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while True:
            if cap:
                success, frame = cap.read()
                if not success:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = cap.read()
                    if not success:
                        time.sleep(0.1)
                        continue
            else:
                frame = self.generate_fallback_image("CAMERA DISCONNECTED")
                time.sleep(1/fps)

            new_frame_time = time.time()
            fps_current = 1 / (new_frame_time - self.prev_frame_time + 0.0001)
            self.prev_frame_time = new_frame_time

            objects_in_frame = 0

            if cap: # Only run YOLO if we have a real frame
                results = self.model.track(frame, persist=True, classes=[0, 39, 67], conf=0.5, verbose=False, imgsz=320)
                local_alert = False

                for r in results:
                    boxes = r.boxes
                    objects_in_frame += len(boxes)
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = self.classNames[cls]
                        track_id = int(box.id[0]) if box.id is not None else -1

                        if currentClass in ["person", "bottle", "cell phone"]:
                            if track_id != -1:
                                colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0)] # Red, Yellow, Green
                                color = colors[track_id % len(colors)]
                            else:
                                color = (0, 255, 0)

                            cvzone.cornerRect(frame, (x1, y1, w, h), l=20, t=3, rt=1, colorR=color, colorC=color)
                            cvzone.putTextRect(frame, f'{currentClass}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorT=(255,255,255), colorR=color, font=cv2.FONT_HERSHEY_PLAIN, offset=5)
                            
                            
                with self.lock:
                    self.current_fps = int(fps_current)
                    self.objects_count = objects_in_frame

            else:
                 with self.lock:
                    self.current_fps = 0
                    self.objects_count = 0
                    self.global_alert = False

            with self.lock:
                self.current_frame = frame.copy()
                if self.is_recording and cap: # only record if we have real video
                    if self.out is None:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        video_path = os.path.join(self.videos_dir, f'output_{timestamp}.avi')
                        self.out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
                    self.out.write(frame)
                elif self.out is not None:
                    self.out.release()
                    self.out = None

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with self.frame_condition:
                    self.latest_frame_bytes = buffer.tobytes()
                    self.frame_condition.notify_all()

    def start(self):
        bg_thread = threading.Thread(target=self.process_video, daemon=True)
        bg_thread.start()

tracker = TrackerApp()
tracker.start()

app = Flask(__name__)

def gen_frames():
    while True:
        with tracker.frame_condition:
            tracker.frame_condition.wait()
            if tracker.latest_frame_bytes is None:
                continue
            frame_data = tracker.latest_frame_bytes
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    with tracker.lock:
        return jsonify({
            'fps': tracker.current_fps,
            'people_count': tracker.objects_count
        })


@app.route('/screenshot', methods=['POST'])
def screenshot():
    with tracker.lock:
        if tracker.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(tracker.screenshots_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, tracker.current_frame)
            return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'status': 'error', 'message': 'No frame available'})

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    with tracker.lock:
        tracker.is_recording = not tracker.is_recording
        state = tracker.is_recording
    return jsonify({'status': 'success', 'is_recording': state})

@app.route('/gallery')
def gallery():
    shots = sorted([f for f in os.listdir(tracker.screenshots_dir) if f.endswith('.jpg')], reverse=True)
    vids = sorted([f for f in os.listdir(tracker.videos_dir) if f.startswith('output_') and f.endswith('.avi')], reverse=True)
    return render_template('gallery.html', screenshots=shots, videos=vids)

@app.route('/media/screenshots/<filename>')
def get_screenshot(filename):
    return send_from_directory(tracker.screenshots_dir, filename)

@app.route('/media/videos/<filename>')
def get_video(filename):
    return send_from_directory(tracker.videos_dir, filename)

@app.route('/delete_media', methods=['POST'])
def delete_media():
    data = request.json
    filename = data.get('filename')
    mtype = data.get('type')
    
    if mtype == 'screenshot':
        filepath = os.path.join(tracker.screenshots_dir, filename)
    elif mtype == 'video':
        filepath = os.path.join(tracker.videos_dir, filename)
    else:
        return jsonify({'status': 'error', 'message': 'Invalid type'})
        
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'File not found'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
