webcam_test.py file: 
    file used to confirm the camera on our devices were working. was used in debugging scenarios when the webcam would not 
    work in demo.py which would indicate something else was causing the issue in the program if it worked in 
    webcam_test.py.
    
    To run this code, type the following command in the terminal:
        python webcam_test.py

training.pkl
    this file is used to store the encoded faces from face_Identification.py. this would then be read into demo.py
    and is used to identify whether a face is known or not. 

face_Identificaion.py
    the code is similar to demo.py except this allows the user to put a name to a face and encode it. The encoded face is then 
    written into training.pkl to be used in demo.py. In order for this code to work, make sure only 1 face is being highlighted. 
    because the code is derived from demo.py, it will highlight all faces it can find. The import statements used are cv2, face_recognition
    dlib, and pickle. these imports are also used in demo.py. 
    if the user does not want to encode their face, they can terminate the program by entering 'q' into the terminal and the webcam will close.

    cv2 import statement is used to turn the webcam on and stream bytes of data into the program. grayscale and RBG scales are later used 
    in the program for different uses. the grayscale increases efficiency when tracking a face and the RGB is used during the encoding
    aspect of the face because we want to capture as much detail possible for accurate identification. because tracking a face does not
    require much detail, grayscale can be used as it just needs to use dlib to identify the characteristics of a face.
    
    dlib is a a package written in c / c++ that we used to identify faces for tracking. dlib already has facial recognition software that
    we used so we decided to utilize code already built to implement our AI project. we specifically used the frontal_face_detector 
    which like the name suggests, only looks at the front of the face to identify faces. dlib has many other options as well which we decided not 
    to use as this best fit our needs. 

    after a face has been identified to be captured as indicated by a box around the face, the user will then press c, where the 
    image will freeze and the user will be prompted to type the name into the terminal. that will then enter the name along with the 
    encoded face into the pkl file which will be read into the demo.py program.

    dlib, cv2, pickle, face_recognition will all need to be installed on your machine to run this program. 
    
    to run this file type the following commands in the terminal:
        python face_Identification.py 

        note: the webcam is set to capture on index 1 and the index might need to be adjusted depending on your computer.

demo.py
    this code uses the same import statements as face_Identification but instead of encoding faces into the pkl file, it will read from
    that file and identify any face that is encoded in the file. if there is not a face known, the program will highlight the face in a 
    yellow box and display the name "unknown". otherwise, it will display the name of the face found if found in the match block and 
    highight the face in a blue box. the user will be able to terminate the demo.py code by entering q, run the face_Identification code and 
    follow the steps stated above to encode their face, and then run the demo and see that they are now identified. the demo will display the number
    of faces found in the terminal along with the names of those found. to end this code, the user has to enter 'q' or ^c in the terminal and the 
    program will end.

    dlib, cv2, pickle, face_recognition will all need to be installed on your machine to run this program. 

    to run this file, type the following commands in the terminal:
        python demo.py

        note: the webcam is set to capture on index 1 and the index might need to be adjusted depending on your computer.
