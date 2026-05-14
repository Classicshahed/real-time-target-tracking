# pyrefly: ignore [missing-import]
import cv2
from ultralytics import YOLO
import cvzone
import math
import os
import time
import random
from collections import defaultdict
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # 1. Ensure screenshots & videos directories exist
    screenshots_dir = "screenshots"
    videos_dir = "videos"
    os.makedirs(screenshots_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # 2. Load the YOLOv8 model (Nano version for real-time speed)
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # Class names for YOLOv8 (Class 0 is 'person')
    classNames = model.names

    # 3. Open Camera (IP or Built-in)
    stream_url_env = os.environ.get("STREAM_URL", "0")
    stream_url = int(stream_url_env) if stream_url_env.isdigit() else stream_url_env

    print(f"Connecting to Camera at: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("IP Camera failed! Falling back to built-in webcam (0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("CRITICAL ERROR: Could not find any camera stream!")
            return

    # Get video properties for the VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30 # Default to 30 if stream doesn't provide it

    # 4. Set up VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_path = os.path.join(videos_dir, f'output_{timestamp}.avi')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    # Variables for Line Crossing Counter
    line_y = frame_height // 2  # A horizontal line in the middle of the screen
    tracker_history = {}        # Stores the last known (cx, cy) position for each ID
    total_count = set()         # Stores unique IDs that have crossed the line
    
    # Dictionary to store the past positions for drawing trails
    track_paths = defaultdict(list)

    print("Started Tracking. Press 'q' to quit, 's' to take a screenshot.")

    # 5. Process video frames in a loop
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Failed to grab frame or stream ended.")
            break

        # Calculate FPS
        new_frame_time = time.time()
        fps_current = 1 / (new_frame_time - prev_frame_time + 0.0001)
        prev_frame_time = new_frame_time

        # Run YOLOv8 tracking on the frame
        # classes=[0] ensures we ONLY detect and track 'person'
        results = model.track(frame, persist=True, classes=[0], conf=0.5, verbose=False, imgsz=320)

        # Draw the virtual crossing line (Red)
        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), 3)

        # Process results and draw advanced UI
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence Score
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # Tracking ID (if available)
                track_id = int(box.id[0]) if box.id is not None else -1

                if currentClass == "person":
                    # --- ADVANCED UI DRAWING ---
                    
                    # Generate a unique color for each tracking ID
                    if track_id != -1:
                        random.seed(track_id)
                        # We use higher values to ensure bright, neon-like colors
                        color = (random.randint(50, 255), random.randint(50, 255), random.randint(100, 255))
                    else:
                        color = (255, 0, 255) # Default purple if no ID

                    # 1. Sleek Corner Bounding Box (cvzone)
                    # rt=0 means no internal rectangle, just corners
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=30, t=5, rt=1,
                                      colorR=color, colorC=color)
                    
                    # 2. Sleek Text Box with ID
                    # We create a display text that includes the ID if it exists
                    display_text = f"ID:{track_id} {currentClass} {conf}" if track_id != -1 else f"{currentClass} {conf}"
                    
                    # Draw a semi-transparent text box
                    cvzone.putTextRect(frame, display_text, (max(0, x1), max(35, y1 - 10)), 
                                       scale=1.5, thickness=2, offset=5, 
                                       colorR=color, colorT=(255, 255, 255))
                                       
                    # 3. Line Crossing & Trajectory Logic
                    cx, cy = x1 + w // 2, y1 + h // 2
                    
                    if track_id != -1:
                        # Draw a small dot at the center of the person
                        cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
                        
                        # --- Trajectory (Path Trail) Drawing ---
                        track_paths[track_id].append((cx, cy))
                        # Keep only the last 30 positions (about 1 second of trail)
                        if len(track_paths[track_id]) > 30:
                            track_paths[track_id].pop(0)
                            
                        # Draw the trail
                        points = track_paths[track_id]
                        for i in range(1, len(points)):
                            # Make the trail slightly thinner than the dot
                            cv2.line(frame, points[i-1], points[i], color, 2)
                        
                        # --- Line Crossing Logic ---
                        if track_id in tracker_history:
                            prev_cx, prev_cy = tracker_history[track_id]
                            
                            # Check if the person crossed the line (moved from above to below, or below to above)
                            if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                                total_count.add(track_id)
                                # Flash the line green for a split second when someone crosses
                                cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 5)
                                
                        # Update the history with the current position
                        tracker_history[track_id] = (cx, cy)

        # Draw FPS counter in the top left corner
        cvzone.putTextRect(frame, f'FPS: {int(fps_current)}', (20, 50), 
                           scale=2, thickness=2, offset=10, 
                           colorR=(0, 0, 0), colorT=(0, 255, 0))
                           
        # Draw the Total Crossed counter below the FPS
        cvzone.putTextRect(frame, f'Total Crossed: {len(total_count)}', (20, 100), 
                           scale=2, thickness=2, offset=10, 
                           colorR=(0, 0, 0), colorT=(0, 255, 255))

        # Write the annotated frame to the output video file
        out.write(frame)

        # Display the annotated frame on screen
        cv2.imshow("Real-Time Target Tracking - Camera", frame)

        # Wait for user input (1 ms delay)
        key = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, break the loop and quit
        if key == ord("q"):
            print("Quitting...")
            break
        
        # If 's' is pressed, save a screenshot
        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved to {filename}")

    # 6. Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Video saved to {video_path}")

if __name__ == "__main__":
    main()
