import face_recognition
import os, sys
import cv2
import numpy as np
import math
from PyQt5.QtCore import QObject, pyqtSignal

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition(QObject):
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    signal1 = pyqtSignal(str)  # Define the signal
    signal2 = pyqtSignal(bool, np.ndarray)  #

    def __init__(self):
        super().__init__()  # Call the __init__ method of the parent class
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            Namesplit= image.split(".")
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(Namesplit[0])
        print(self.known_face_names)

    def update_data(self):
        self.signal1.emit("hello1")


    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)

                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Create the frame with the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Display the resulting image
            #cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break
              # Emit the signal with ret and frame values
            self.signal2.emit(ret, frame)
        # Release handle to the webcam
        #video_capture.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        # Capture frame-by-frame from the webcam or video file
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()

        # Resize the frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []

        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = '???'

            # Calculate the shortest distance to face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append((name, confidence))

            # Draw rectangle and display name on the frame
            for (top, right, bottom, left), (name, confidence) in zip(self.face_locations, self.face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with name and confidence below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, f"{name} ({confidence})", (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        #return ret, frame

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
