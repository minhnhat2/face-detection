import face_recognition
import cv2
import time

# Load the reference face image and extract its features
ref_image = face_recognition.load_image_file("nhat.png")
ref_encoding = face_recognition.face_encodings(ref_image)[0]

# Start capturing frames from the camera
cap = cv2.VideoCapture(0)
start_time = time.time()
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    # Find any detected faces in the frame and extract their features
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Compare the features of each detected face with the reference face
        for face_encoding in face_encodings:
            distance = face_recognition.face_distance([ref_encoding], face_encoding)
            if distance < 0.6:  # Adjust the threshold as needed
                print("Face match found!")
                # Insert your desired code here to continue running
                break
    
    # Check if the time limit has been reached
    elapsed_time = time.time() - start_time
    if elapsed_time > 60:
        print("Time limit reached. Exiting program.")
        break
    
    # Display the current frame (optional)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
