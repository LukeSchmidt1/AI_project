import cv2
import dlib
import pickle
import face_recognition

training_file = 'training.pkl'

detector = dlib.get_frontal_face_detector()  # Pre-trained face detection
face_encodings = []
name_lst = []

cap = cv2.VideoCapture(1)  # Adjust camera index if necessary

if not cap.isOpened():
    print("Could not open webcam")
    exit()

face_prompted = False  # Flag to track if a prompt has been shown

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(gray)

    if faces:  # If at least one face is detected
        if not face_prompted:  # Show the prompt only once
            print("Wait for face to be highlighted, then press 'c' to capture or 'q' to quit")
            face_prompted = True

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_location = [(y, x + w, y + h, x)]
            face_encode = face_recognition.face_encodings(rgb_frame, face_location)

            if face_encode:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # Capture face
                    name = input("Enter name: ")
                    face_encodings.append(face_encode[0])
                    name_lst.append(name)
                    print("Face recorded successfully")
                    face_prompted = False  # Reset prompt flag after successful capture
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Exit the for loop after capturing
                elif key == ord('q'):  # Quit program
                    print("Exiting program.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
    else:
        face_prompted = False  # Reset prompt flag if no faces are detected

    cv2.imshow("Tracking - press q to exit", frame)

    # Check if 'q' is pressed at any point, to exit the loop gracefully
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Allow user to quit
        print("Exiting program.")
        break
# Save the data to the training file if faces are recorded without writing over current data
if face_encodings:
    try:
        with open(training_file, 'rb') as file:
            existing_data = pickle.load(file)
    except(Exception, EOFError):
        existing = {'face encodings': [], 'names': []}

existing_data['face encodings'].extend(face_encodings)
existing_data['names'].extend(name_lst)

with open(training_file, 'wb') as file:
    pickle.dump(existing_data, file)
    print('successfully updated to the file')
