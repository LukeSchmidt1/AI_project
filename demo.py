import cv2
import dlib
import pickle
import face_recognition

try:
    with open('training.pkl', 'rb') as file:    # reads training.pkl which holds all the known faces / names
        data = pickle.load(file)
        encodings = data['face encodings']
        names = data['names']
except(Exception, EOFError):
    encodings = []
    names = []

detector = dlib.get_frontal_face_detector() # opens pre trained face detection 

cap = cv2.VideoCapture(1)   # opens default webcam

if not cap.isOpened():  # tries to open webcam. fails if unable to 
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

    for face in faces:      # draws rectangles around face and tracks it
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rectangle_color = (0, 255, 255)     # face is automatically boxed with yellow with unknown name displayed
        location = [(y, x+w, y+h, x)]       # location of face
        face_encode = face_recognition.face_encodings(rgb_frame, location)

        if face_encode:     # if there is a face found, check if it is known or not
            encoding = face_encode[0]
            matches = face_recognition.compare_faces(encodings, encoding, tolerance=.6) # looks for a match in the file
            name = "Unknown face"   # default name is Unknown

            if True in matches:     # if a face is found, display the name
                index = matches.index(True) 
                name = names[index]
                rectangle_color = (255,0, 0)
                
        cv2.rectangle(frame, (x,y), (x+w, y+h), rectangle_color, 2)
        cv2.putText(frame, name, (x,y - 10), cv2.FONT_HERSHEY_DUPLEX, .8, (255, 255, 255), 2)   # displays the name at the top left of the box
        print(f'Number detected: {len(faces)}\tFaces detected: {name}') # prints the name detected

    cv2.imshow("Facial Tracking, Press q to exit:", frame)  # shows facial recognition until user enters q

    if cv2.waitKey(1) & 0xFF == ord('q'):      # if user enters q, loop ends
        break


cap.release()       # closes webcam, destroys windows
cv2.destroyAllWindows()
