import logging
from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import base64
import traceback
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("server.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load the trained LSTM model.
try:
    model = load_model('asl_model.h5')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise e

# Define the class names.
class_names = ['Hello', 'Thank You', 'Yes', 'No', 'I Love You']  # Update with your classes.

app = Flask(__name__)

def predict_sign(landmarks_sequence):
    """
    Predict the sign language gesture based on a sequence of hand landmarks.
    """
    try:
        # Preprocess landmarks as required by your model.
        sequence = np.array(landmarks_sequence)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension.

        # Predict using your model.
        prediction = model.predict(sequence)
        class_id = np.argmax(prediction)
        sign = class_names[class_id]
        return sign
    except Exception as e:
        logger.error("Error in predict_sign: %s", e)
        traceback.print_exc()
        return None

# Store sequences of landmarks.
landmarks_buffer = []
SEQUENCE_LENGTH = 30  # Number of frames to consider for prediction.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the frame from the client.
        data = request.get_json()
        img_str = data['image']

        # Decode the image.
        img_bytes = base64.b64decode(img_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process the image and find hands.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates.
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_buffer.append(landmarks)

                # Maintain buffer size.
                if len(landmarks_buffer) > SEQUENCE_LENGTH:
                    landmarks_buffer.pop(0)

                # Predict when we have enough frames.
                if len(landmarks_buffer) == SEQUENCE_LENGTH:
                    sign = predict_sign(landmarks_buffer)
                    landmarks_buffer.clear()  # Clear buffer after prediction.
                    if sign:
                        logger.info("Predicted sign: %s", sign)
                        return jsonify({'sign': sign})
        else:
            landmarks_buffer.clear()  # Clear buffer if no hands detected.

        return jsonify({'sign': ''})
    except Exception as e:
        logger.error("Error in /predict endpoint: %s", e)
        traceback.print_exc()
        return jsonify({'error': 'Server error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
