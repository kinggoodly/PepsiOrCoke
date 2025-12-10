import cv2
import os

save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = f"{save_dir}/img_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
