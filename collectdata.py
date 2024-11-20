import os
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)
directory = 'C:/Users/Alishna/Desktop/Vs code/SignLang/Image/'

# Ensure the directory structure exists
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

while True:
    _, frame = cap.read()
    count = {chr(i): len(os.listdir(os.path.join(directory, chr(i)))) for i in range(ord('A'), ord('Z') + 1)}
    
    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])
    frame = frame[40:400, 0:300]
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF in [ord(chr(i)) for i in range(ord('a'), ord('z') + 1)]:
        letter = chr(interrupt & 0xFF).upper()
        image_path_jpg = os.path.join(directory, letter, f"{letter}({count[letter]}).jpg")
        image_path_jpeg = os.path.join(directory, letter, f"{letter}({count[letter]}).jpeg")
        
        cv2.imwrite(image_path_jpg, frame)
        cv2.imwrite(image_path_jpeg, frame)
        
        print(f"Saved {image_path_jpg} or {image_path_jpeg}")

    if interrupt & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()