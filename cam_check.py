import cv2

for i in range(5):
    print(f"Testing camera {i}...")
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
    else:
        print(f"Camera {i} not found")
    cap.release()
