import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    """Detects OK sign and Thumbs Up gesture"""
    landmarks = hand_landmarks.landmark

    # Get finger tip and index finger MCP coordinates
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    thumb_mcp = landmarks[2]

    # Calculate distances
    thumb_index_dist = np.linalg.norm(
        [thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]
    )

    thumb_mcp_index_mcp_dist = np.linalg.norm(
        [thumb_mcp.x - index_mcp.x, thumb_mcp.y - index_mcp.y]
    )

    # Detect OK Sign (Thumb tip close to Index tip)
    if thumb_index_dist < 0.05:
        return "OK"

    # Detect Thumbs Up (Thumb tip higher than all other fingers)
    if (thumb_tip.y < index_tip.y and 
        thumb_tip.y < landmarks[12].y and 
        thumb_tip.y < landmarks[16].y and 
        thumb_tip.y < landmarks[20].y):
        return "Thumbs Up"

    return None  # No gesture detected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(rgb_frame)

    gesture_text = ""

    # Draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            gesture_text = detect_gesture(hand_landmarks)

    # Display gesture text on screen
    if gesture_text:
        cv2.putText(frame, gesture_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Hand Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
