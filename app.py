import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load your trained model
try:
    model = load_model("asl_hand_model.h5")
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to extract landmarks
def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        normalized = np.array(landmarks) - np.tile([wrist.x, wrist.y, wrist.z], 21)
        return normalized
    else:
        return None

# Start webcam
cap = cv2.VideoCapture(0)

# Variables for word building
current_word = ""
last_detection_time = time.time()
last_prediction_time = time.time()

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        break

    frame_height, frame_width, _ = frame.shape

    # Extract landmarks
    landmarks = extract_landmarks(frame)

    current_time = time.time()

    # Only predict every 2 seconds
    if landmarks is not None and current_time - last_prediction_time > 2:
        prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
        confidence = np.max(prediction)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index] if confidence > 0.7 else "..."
        last_prediction_time = current_time

        if predicted_label not in ["...", "nothing", "space", "del"]:
            current_word += predicted_label
        elif predicted_label == "del":
            current_word = current_word[:-1]

        # Update detection time
        last_detection_time = current_time

    # Reset word if no hand detected for 4 seconds
    elif landmarks is None and current_time - last_detection_time > 4 and current_word:
        current_word = ""
        last_detection_time = current_time

    # Overlay the word at the bottom center
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(current_word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 50

    # Draw yellow background and black text
    cv2.rectangle(frame, (text_x - 20, text_y - 40), (text_x + text_size[0] + 20, text_y + 10), (0, 255, 255), -1)  # Yellow box
    cv2.putText(frame, current_word, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Show confidence bar (only if we made a prediction recently)
    if landmarks is not None:
        confidence = np.max(model.predict(np.expand_dims(landmarks, axis=0), verbose=0))
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (frame_width - 320, 50), (frame_width - 20, 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (frame_width - 320, 50), (frame_width - 320 + bar_width, 70), (0, 255, 0), -1)
        cv2.putText(frame, f"{int(confidence * 100)}% Confident", (frame_width - 320 + 10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("ASL Sign Interpreter", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()