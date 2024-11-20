import os
import cv2
import numpy as np
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define actions and other variables
actions = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])
DATA_PATH = 'MP_Data'
no_sequences = 30
sequence_length = 30

# Create directories for storing data
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read image in either .jpg or .jpeg format
                image_path_jpg = os.path.join('Image', action, f'{action}({sequence}).jpg')
                image_path_jpeg = os.path.join('Image', action, f'{action}({sequence}).jpeg')

                if os.path.exists(image_path_jpg):
                    frame = cv2.imread(image_path_jpg)
                elif os.path.exists(image_path_jpeg):
                    frame = cv2.imread(image_path_jpeg)
                else:
                    print(f"Image not found: {image_path_jpg} or {image_path_jpeg}")
                    continue
                
                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic for first frame
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
