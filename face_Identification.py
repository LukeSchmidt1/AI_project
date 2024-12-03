import cv2
import dlib
import pickle
import face_recognition

training_file = 'training.pkl'

detector = dlib.get_frontal_face_detector()  # Pre-trained face detection
face_encodings = []
name_lst = []

cap = cv2.VideoCapture(1)  # Adjust camera index if necessary
                           # might fail if index is off or connect to other devices

if not cap.isOpened():      # attempts to open webcam. fails if cant open
    print("Could not open webcam")
    exit()

face_prompted = False  # Flag to track if a prompt has been shown
                        # used to prompt the user once inside the loop

while cap.isOpened():
    ret, frame = cap.read()    # reads from the camera and will loop until user presses q or c to close it
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # converts to grayscale reducing complexity for tracking
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # converts to RGB scale
    faces = detector(gray)      # only accepts grayscale to detect the faces

    if faces:  # If at least one face is detected
        if not face_prompted:  # Show the prompt only once
            print("Wait for face to be highlighted, then press 'c' to capture or 'q' to quit")
            face_prompted = True

        for face in faces:      # finds the face and puts a box around it
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_location = [(y, x + w, y + h, x)]
            face_encode = face_recognition.face_encodings(rgb_frame, face_location)

            if face_encode:     # if there is a face found. encode it or quit depending on user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # Capture image of face
                    name = input("Enter name: ")    # get name of person
                    face_encodings.append(face_encode[0])   # add to the list of faces to encode
                    name_lst.append(name)       
                    print("Face recorded successfully")
                    face_prompted = False  # Reset prompt flag after successful capture
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Exit the for loop after capturing
                elif key == ord('q'):  # Quit program
                    print("Exiting program.")
                    cap.release()   # stop streaming from webcam
                    cv2.destroyAllWindows()     # destroys all the windows
                    exit()
    else:
        face_prompted = False  # Reset prompt flag if no faces are detected

    cv2.imshow("Tracking - press q to exit", frame)     # displays the facial recognition

    # Check if 'q' is pressed at any point, to exit the loop gracefully
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # checks user input if they entered q, quit if true
        print("Exiting program.")
        break

# Save the data to the training file if faces are recorded without writing over current data
if face_encodings:          # grabs current data from pkl file
    try:
        with open(training_file, 'rb') as file:
            existing_data = pickle.load(file)
    except(Exception, EOFError):
        existing_data = {'face encodings': [], 'names': []}  # if no data, creates empty list

existing_data['face encodings'].extend(face_encodings)      #adds to the list of names / faces
existing_data['names'].extend(name_lst)

with open(training_file, 'wb') as file:     # writes to the pkl file with either empty list if no values
    pickle.dump(existing_data, file)        # or writes the names and faces that exist + the one added
    print('successfully updated to the file')
