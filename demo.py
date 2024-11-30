import cv2
import dlib
import pickle
import face_recognition

with open('training.pkl', 'rb') as file:
    data = pickle.load(file)
    encodings = data['face encodings']
    names = data['names']

detector = dlib.get_frontal_face_detector() # opens pre trained face detection 

cap = cv2.VideoCapture(1)   # opens default webcam

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while cap.isOpened():   # infinite loop if webcam opens
    ret, frame = cap.read() # reads frames, ret is boolean indicating successfully read
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converts frame to grayscale for faster processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(gray)  # inserts grayscale into pre trained detection and returns list of detected faces

    print(f"Faces detected: {len(faces)}") 

    for face in faces:      # draws rectangles around face and tracks it
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        location = [(y, x+w, y+h, x)]
        face_encode = face_recognition.face_encodings(rgb_frame, location)

        if face_encode:
            encoding = face_encode[0]
            matches = face_recognition.compare_faces(encodings, encoding, tolerance=.6)
            name = "Unknown face"

            if True in matches:
                index = matches.index(True)
                name = names[index]

        cv2.putText(frame, name, (x,y - 10), cv2.FONT_HERSHEY_PLAIN, .8, (255, 255, 255), 2)
        print(f'Face detected: {name}')

    cv2.imshow("Facial Tracking, Press q to exit:", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):      # if user enters q, loop ends
        break


cap.release()       # closes webcam 
cv2.destroyAllWindows()
