# client.py
import cv2
import asyncio
import websockets
import logging
import sys
import signal
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Replace with your video source (0 = Webcam, or RTMP URL)
video_source = 0  # Or: 'rtmp://your-stream-url'

# Global flag for graceful shutdown
shutdown_flag = asyncio.Event()

def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()

async def send_frame(websocket, frame):
    # Encode the frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Send the frame to the server
    await websocket.send(frame_bytes)
    logger.debug("Frame sent to server.")

async def receive_detection_results(websocket):
    try:
        detection_results = await asyncio.wait_for(websocket.recv(), timeout=1.0)
        if detection_results:
            try:
                detection_data = json.loads(detection_results)
                logger.info(f"Detection results: {detection_data}")
                return detection_data
            except json.JSONDecodeError as json_error:
                logger.warning(f"Received invalid JSON data from server: {json_error}")
        else:
            logger.warning("Received empty response from server.")
    except asyncio.TimeoutError:
        logger.debug("Timeout waiting for server response.")
    except Exception as e:
        logger.error(f"Error receiving detection results: {e}")
    return None

async def send_video():
    try:
        # Connect to the server
        async with websockets.connect('ws://localhost:9002') as websocket:
            logger.info("Connected to server.")
            
            # Try different backends for video capture
            backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_DSHOW]
            cap = None
            
            for backend in backends:
                cap = cv2.VideoCapture(video_source, backend)
                if cap.isOpened():
                    logger.info(f"Successfully opened video source with backend: {backend}")
                    break
                else:
                    cap.release()

            if not cap or not cap.isOpened():
                logger.error(f"Failed to open video source: {video_source}")
                return

            # Create a named window before the loop
            cv2.namedWindow("Client Video", cv2.WINDOW_NORMAL)

            try:
                while not shutdown_flag.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read a frame from the video source.")
                        continue  # Try to read the next frame instead of breaking

                    # Create a copy of the original frame
                    original_frame = frame.copy()

                    # Send the frame to the server
                    try:
                        await send_frame(websocket, frame)
                        
                        # Receive detection results from the server
                        detection_data = await receive_detection_results(websocket)
                        
                        if detection_data:
                            logger.info(f"Received detection data: {detection_data}")
                            # Draw bounding boxes and labels on the frame
                            for detection in detection_data:
                                bbox = detection.get('bbox', [])
                                label = detection.get('label', 'Unknown')
                                confidence = detection.get('confidence', 0.0)
                                color = (0, 255, 0)  # Green color for bounding box
                                
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = map(int, bbox)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    
                                    label_text = f"{label} {confidence:.2f}"
                                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                                else:
                                    logger.warning(f"Invalid bounding box format: {bbox}")
                        else:
                            logger.debug("No detection data received")
                        
                        # Combine original and detection frames side by side
                        combined_frame = np.hstack((original_frame, frame))
                        
                        # Display the combined frame
                        cv2.imshow("Client Video", combined_frame)
                        
                        # Check for 'q' key press to exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logger.info("Client exiting on user request.")
                            break
                        
                    except websockets.exceptions.ConnectionClosed:
                        logger.error("WebSocket connection closed unexpectedly.")
                        break
                    except Exception as e:
                        logger.error(f"Error in video processing loop: {e}")
                        continue

                    if cv2.getWindowProperty("Client Video", cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Client Video window closed. Exiting...")
                        break

                # Add log line to show why camera feed window does not show
                logger.info(f"Camera feed window visibility: {cv2.getWindowProperty('Client Video', cv2.WND_PROP_VISIBLE)}")

            finally:
                logger.info("Releasing camera and closing windows...")
                cap.release()
                cv2.destroyAllWindows()
    except Exception as e:
        logger.critical(f"Client encountered an error: {e}")

async def main():
    # Set up signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal_handler)
    
    try:
        await send_video()
    except Exception as e:
        logger.critical(f"Unhandled error: {e}")
    finally:
        logger.info("Client shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Client interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled error: {e}")
        sys.exit(1)
    finally:
        # Register a signal handler for SIGTSTP (Ctrl+Z)
        def handle_sigtstp(signum, frame):
            logger.info("Received SIGTSTP (Ctrl+Z). Releasing camera resources...")
            if 'cap' in globals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera resources released. Exiting...")
            sys.exit(0)

        signal.signal(signal.SIGTSTP, handle_sigtstp)

        # Add a warning message about AVCaptureDeviceTypeExternal deprecation
        logger.warning("AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras. Please use AVCaptureDeviceTypeContinuityCamera and add NSCameraUseContinuityCameraDeviceType to your Info.plist.")
