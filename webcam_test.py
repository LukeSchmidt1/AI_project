import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
else:
    print("Webcam opened successfully")
cv2.show()

cap.release()