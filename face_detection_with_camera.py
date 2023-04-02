import cv2
import face_recognition

# Load known face images and encodings

#The face_recognition library uses these known face encodings to compare them to the face encodings of new faces that are detected in the video stream. If a match is found between a new face and one of the known face encodings, the script identifies the person as the corresponding known face.
elon_image = face_recognition.load_image_file("faces/elon.png")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

elsa_image = face_recognition.load_image_file("faces/elsa.png")
elsa_encoding = face_recognition.face_encodings(elsa_image)[0]

nhat_image = face_recognition.load_image_file("faces/nhat-dep-trai.png")
nhat_encoding = face_recognition.face_encodings(nhat_image)[0]

rose_image = face_recognition.load_image_file("faces/rose.png")
rose_encoding = face_recognition.face_encodings(rose_image)[0]

# Create arrays of known face encodings and names
known_face_encodings = [
    elon_encoding,
    elsa_encoding,
    nhat_encoding,
    rose_encoding
]
known_face_names = [
    "Elon Musk",
    "Elsa",
    "Nhat Dep Trai",
    "Rose"
]

# Start video capture from default camera
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize the frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
