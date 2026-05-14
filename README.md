# Real-Time Target Tracking & Analytics

An AI-driven real-time video analytics system built with Python, Flask, YOLOv8, and OpenCV. It tracks individuals, counts line crossings, and features a SIFT-based fraud detection system that alerts you when a registered suspect enters the camera frame.

## Prerequisites
Before you start, ensure you have the following installed on your laptop and phone:
1. **Python 3.8+** installed on your laptop.
2. **DroidCam App** installed on your smartphone (available on iOS App Store & Google Play Store).
3. Both your phone and your laptop must be connected to the **same Wi-Fi network**.

## 🚀 Setup Instructions

### 1. Install Dependencies
Open your terminal (or Command Prompt/PowerShell), navigate to the project directory, and run the following command to install all the required Python libraries:

```bash
pip install -r requirements.txt
```

### 2. Configure Your Camera
This project uses your smartphone as a wireless IP camera via DroidCam.

1. Open the **DroidCam app** on your phone.
2. The app will display a **WiFi IP** (e.g., `192.168.0.104`) and a **Port** (usually `4747`).
3. Open the `.env` file in the project folder.
4. Update the `STREAM_URL` to match your phone's IP exactly:
   ```text
   STREAM_URL=http://192.168.0.104:4747/video
   ```

*(Note: If your router restarts or you disconnect from the Wi-Fi, your phone's IP might change. You will need to update the `.env` file whenever the IP changes.)*

### 3. Run the Application

You have two ways to run this project: the Web Dashboard or the Desktop Window.

#### Option A: Web Dashboard (Recommended)
This runs a local Flask server that provides a beautiful web interface.
```bash
python app.py
```
* After running the command, open your web browser and go to: **http://127.0.0.1:5000**
* From the dashboard, you can view the live feed, record video, take screenshots, upload fraud suspect images, and access the media gallery.

#### Option B: Simple Desktop Window
If you just want a quick, bare-bones window to view the tracking without the web dashboard:
```bash
python main.py
```
* A desktop window will pop up.
* Press `s` on your keyboard to save a screenshot.
* Press `q` to quit the application and save the video.

## 📂 Project Structure
* `app.py`: The Flask Web Server and background AI thread.
* `main.py`: The standalone desktop script.
* `yolov8n.pt`: The lightweight YOLO AI model (downloads automatically if missing).
* `.env`: Configuration file containing your camera URL.
* `requirements.txt`: The list of Python packages required.
* `templates/`: Contains the HTML layout for the dashboard (`index.html`) and gallery (`gallery.html`).
* `videos/`: Directory where your recorded `.avi` clips are automatically saved.
* `screenshots/`: Directory where your `.jpg` screenshots are saved.
