import logging
import cv2
import numpy as np
import requests
import base64
import pyttsx3
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("client.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

# Initialize Text-to-Speech engine.
engine = pyttsx3.init()

# Open the webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open webcam")
    exit()

prev_sign = ""

while True:
    try:
        success, frame = cap.read()
        if not success:
            logger.warning("Ignoring empty camera frame.")
            continue

        # Encode the frame as JPEG.
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # Send the frame to the server.
        response = requests.post('http://localhost:5000/predict', json={'image': img_base64}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            sign = data.get('sign', '')

            # If sign changed, speak it out.
            if sign and sign != prev_sign:
                logger.info("Recognized sign: %s", sign)
                engine.say(sign)
                engine.runAndWait()
                prev_sign = sign

            # Display the sign on the screen.
            cv2.putText(frame, 'Sign: ' + sign, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            logger.error("Server returned status code %s", response.status_code)
            cv2.putText(frame, 'Error: Server issue', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the image.
        cv2.imshow('Hand Sign Recognition Client', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    except requests.exceptions.RequestException as e:
        logger.error("Network error: %s", e)
        traceback.print_exc()
        cv2.putText(frame, 'Error: Network issue', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Hand Sign Recognition Client', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        traceback.print_exc()
        break

cap.release()
cv2.destroyAllWindows()
