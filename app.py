from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
from ultralytics import YOLO
import cvzone
import math
import os
import time
import random
from collections import defaultdict
import threading
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Globals & Setup
screenshots_dir = "screenshots"
videos_dir = "videos"
os.makedirs(screenshots_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)

print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
classNames = model.names

is_recording = False
out = None
current_frame = None
latest_frame_bytes = None
frame_condition = threading.Condition()
lock = threading.Lock()

line_y = 240
tracker_history = {}
total_count = set()
track_paths = defaultdict(list)
prev_frame_time = 0

# Fraud Detection Globals
sift = cv2.SIFT_create()
fraud_kp = None
fraud_des = None
fraud_ids = set()
checked_ids = set()
global_alert = False

def process_video():
    global current_frame, latest_frame_bytes, is_recording, out, prev_frame_time, line_y, global_alert
    
    stream_url_env = os.environ.get("STREAM_URL", "0")
    stream_url = int(stream_url_env) if stream_url_env.isdigit() else stream_url_env

    print(f"Connecting to Camera at: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("Camera failed to connect over IP! Falling back to built-in webcam (0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam also failed! Falling back to sample_video.mp4...")
            cap = cv2.VideoCapture("sample_video.mp4")
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: fps = 30
    if frame_height > 0: line_y = frame_height // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
        new_frame_time = time.time()
        fps_current = 1 / (new_frame_time - prev_frame_time + 0.0001)
        prev_frame_time = new_frame_time

        results = model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False, imgsz=320)
        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 3)
        
        local_alert = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                track_id = int(box.id[0]) if box.id is not None else -1

                if currentClass == "person":
                    is_fraud = False
                    
                    with lock:
                        local_fraud_des = fraud_des
                        
                    if local_fraud_des is not None:
                        if track_id != -1 and track_id in fraud_ids:
                            is_fraud = True
                        elif track_id != -1 and track_id not in checked_ids:
                            x1_c, y1_c = max(0, x1), max(0, y1)
                            x2_c, y2_c = min(frame_width, x2), min(frame_height, y2)
                            if x2_c - x1_c > 20 and y2_c - y1_c > 20:
                                person_crop = frame[y1_c:y2_c, x1_c:x2_c]
                                gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                                kp_crop, des_crop = sift.detectAndCompute(gray_crop, None)
                                
                                if des_crop is not None and len(des_crop) > 0:
                                    bf = cv2.BFMatcher()
                                    try:
                                        matches = bf.knnMatch(local_fraud_des, des_crop, k=2)
                                        good = []
                                        for match_set in matches:
                                            if len(match_set) == 2:
                                                m, n = match_set
                                                if m.distance < 0.75 * n.distance:
                                                    good.append(m)
                                                    
                                        if len(good) >= 12: 
                                            is_fraud = True
                                            with lock:
                                                fraud_ids.add(track_id)
                                    except Exception:
                                        pass
                                        
                            with lock:
                                checked_ids.add(track_id)
                    
                    if is_fraud:
                        color = (0, 0, 255) # RED
                        local_alert = True
                        display_text = f"🚨 FRAUD ID:{track_id} 🚨 {conf}" if track_id != -1 else f"🚨 FRAUD 🚨 {conf}"
                    else:
                        if track_id != -1:
                            random.seed(track_id)
                            color = (random.randint(50, 255), random.randint(50, 255), random.randint(100, 255))
                            display_text = f"ID:{track_id} {currentClass} {conf}"
                        else:
                            color = (255, 0, 255)
                            display_text = f"{currentClass} {conf}"

                    cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, rt=1, colorR=color, colorC=color)
                    cvzone.putTextRect(frame, display_text, (max(0, x1), max(35, y1 - 10)), scale=1.5, thickness=2, offset=5, colorR=color, colorT=(255, 255, 255))
                                       
                    cx, cy = x1 + w // 2, y1 + h // 2
                    
                    if track_id != -1:
                        cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
                        track_paths[track_id].append((cx, cy))
                        if len(track_paths[track_id]) > 30: track_paths[track_id].pop(0)
                            
                        points = track_paths[track_id]
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i-1], points[i], color, 2)
                        
                        if track_id in tracker_history:
                            prev_cx, prev_cy = tracker_history[track_id]
                            if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                                total_count.add(track_id)
                                cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 5)
                                
                        tracker_history[track_id] = (cx, cy)
                        
        with lock:
            global_alert = local_alert

        cvzone.putTextRect(frame, f'FPS: {int(fps_current)}', (20, 50), scale=2, thickness=2, offset=10, colorR=(0, 0, 0), colorT=(0, 255, 0))
        cvzone.putTextRect(frame, f'Total Crossed: {len(total_count)}', (20, 100), scale=2, thickness=2, offset=10, colorR=(0, 0, 0), colorT=(0, 255, 255))

        with lock:
            current_frame = frame.copy()
            if is_recording:
                if out is None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    video_path = os.path.join(videos_dir, f'output_{timestamp}.avi')
                    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
                out.write(frame)
            elif out is not None:
                out.release()
                out = None

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            with frame_condition:
                latest_frame_bytes = buffer.tobytes()
                frame_condition.notify_all()

# Start background thread
bg_thread = threading.Thread(target=process_video, daemon=True)
bg_thread.start()

def gen_frames():
    while True:
        with frame_condition:
            frame_condition.wait()
            if latest_frame_bytes is None:
                continue
            frame_data = latest_frame_bytes
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_fraud', methods=['POST'])
def upload_fraud():
    global fraud_kp, fraud_des, fraud_ids, checked_ids, global_alert
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        with lock:
            fraud_kp, fraud_des = kp, des
            fraud_ids.clear()
            checked_ids.clear()
            global_alert = False
            
        return jsonify({'status': 'success', 'message': 'Fraud suspect registered!'})
    return jsonify({'status': 'error', 'message': 'Invalid image'})

@app.route('/alert_status')
def alert_status():
    with lock:
        return jsonify({'alert': global_alert})

@app.route('/screenshot', methods=['POST'])
def screenshot():
    with lock:
        if current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, current_frame)
            return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'status': 'error', 'message': 'No frame available'})

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    global is_recording
    with lock:
        is_recording = not is_recording
        state = is_recording
    return jsonify({'status': 'success', 'is_recording': state})

@app.route('/gallery')
def gallery():
    shots = sorted([f for f in os.listdir(screenshots_dir) if f.endswith('.jpg')], reverse=True)
    vids = sorted([f for f in os.listdir(videos_dir) if f.startswith('output_') and f.endswith('.avi')], reverse=True)
    return render_template('gallery.html', screenshots=shots, videos=vids)

@app.route('/media/screenshots/<filename>')
def get_screenshot(filename):
    return send_from_directory(screenshots_dir, filename)

@app.route('/media/videos/<filename>')
def get_video(filename):
    return send_from_directory(videos_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
