# server.py
import cv2
import asyncio
import numpy as np
import torch
from ultralytics import YOLO
from gtts import gTTS
import os
import logging
from flask import Flask
from flask_socketio import SocketIO
import websockets
import signal
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")

# Load the YOLOv11 model
try:
    logger.info("Loading YOLOv11 model...")
    model = YOLO('yolo11n.pt')  # Replace with the path to your YOLOv11 model
except Exception as e:
    logger.error(f"Failed to load YOLOv11 model: {e}")
    raise

# Global variables
server = None
cap = None

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

async def handle_video_frame(websocket, data):
    logger.info("Received video frame from client")
    
    try:
        # Decode the incoming frame
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning("Received an invalid frame.")
            return

        # Perform YOLOv11 inference
        results = model(frame)
        predictions = results[0].boxes.data

        detection_results = []
        if len(predictions) > 0:
            for pred in predictions:
                x1, y1, x2, y2, conf, class_id = pred
                sign_detected = results[0].names[int(class_id)]
                detection_results.append({
                    'label': sign_detected,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
            
            logger.info(f"Detected Signs: {[res['label'] for res in detection_results]}")
            logger.info(f"Sending detection results to client: {json.dumps(detection_results)}")

            # Convert the detected sign to speech (keeping the first detection for audio)
            tts = gTTS(text=detection_results[0]['label'], lang='en')
            tts.save("sign_audio.mp3")
            os.system("mpg321 sign_audio.mp3")  # Play the audio
        else:
            logger.info("No sign detected in the frame.")

        # Send detection results back to the client
        await websocket.send(json.dumps(detection_results))

    except cv2.error as opencv_err:
        handle_opencv_exception(opencv_err)
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")

@socketio.on_error_default
def handle_error(e):
    logger.error(f"SocketIO Error: {e}")

async def websocket_handler(websocket, path):
    try:
        logger.info(f"New WebSocket connection: {path}")
        async for message in websocket:
            await handle_video_frame(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}. Shutting down...")
    if isinstance(server, websockets.server.WebSocketServer):
        server.close()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    try:
        logger.info("Starting server...")
        loop = asyncio.get_event_loop()
        server = websockets.serve(websocket_handler, "0.0.0.0", 9002)
        loop.run_until_complete(server)
        loop.run_forever()
    except Exception as e:
        logger.critical(f"Server failed to start: {e}")
        raise
    finally:
        # Clean up resources
        if isinstance(server, websockets.server.WebSocketServer):
            server.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Server shutting down...")

def handle_opencv_exception(e):
    logger.error(f"OpenCV Error: {str(e)}")
    if "Unknown C++ exception" in str(e):
        logger.error("This error might be due to a memory issue or incompatible OpenCV version.")
        logger.info("Attempting to release and reinitialize video capture...")
        global cap
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(0)  # Reinitialize the video capture
        if not cap.isOpened():
            logger.error("Failed to reinitialize video capture. Please check your camera connection.")
    else:
        logger.error("Unhandled OpenCV error. Please check your OpenCV installation and camera drivers.")

# Add a periodic task to check and reinitialize video capture if needed
async def check_video_capture():
    global cap
    while True:
        await asyncio.sleep(60)  # Check every minute
        if cap is None or not cap.isOpened():
            logger.info("Video capture is not open. Attempting to reinitialize...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to reinitialize video capture.")
            else:
                logger.info("Video capture reinitialized successfully.")

# Add the periodic task to the event loop
loop = asyncio.get_event_loop()
loop.create_task(check_video_capture())
