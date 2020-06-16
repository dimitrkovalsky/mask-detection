import pickle
from os import path
from pathlib import Path

import face_recognition


class Recognizer:
    def __init__(self, directory='storage/', dump_name="encodedFaces.pkl"):
        self.directory = directory
        self.dump_name = dump_name
        self.faceEncodings = self.try_load_encodings()

    def save_obj(self, obj):
        with open(self.get_dump_path(), 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self):
        with open(self.get_dump_path(), 'rb') as f:
            return pickle.load(f)

    def get_dump_path(self):
        return self.directory + self.dump_name

    def encode_faces(self):
        file_list = Path("people").glob("*.*")
        faceEncodings = {}
        for image_path in file_list:
            # Load an image to check
            unknown_image = face_recognition.load_image_file(image_path)

            # Get the location of faces and face encodings for the current image
            face_encodings = face_recognition.face_encodings(unknown_image)
            faceEncodings[image_path] = face_encodings
            print(image_path, "encoded as ", face_encodings)
        self.save_obj(faceEncodings)
        print("Encoded all faces from directory")
        return faceEncodings

    def load_encodings(self):
        faceEncodings = self.load_obj()
        print("Loaded ", len(faceEncodings), "people")
        return faceEncodings

    # todo: optimize using last n encodings instead of all array
    def recognize_person(self, face_encoding):
        best_face_distance = 1.0
        best_face_name = "unknown"
        if face_encoding is None:
            return best_face_name
        for (fileName, encoded) in self.faceEncodings.items():
            face_distance = face_recognition.face_distance(face_encoding, encoded)[0]
            if face_distance < best_face_distance:
                best_face_distance = face_distance
                # Extract a copy of the actual face image itself so we can display it
                best_face_name = fileName.stem

        return best_face_name

    def try_load_encodings(self):
        if path.exists(self.get_dump_path()):
            return self.load_encodings()
        else:
            return self.encode_faces()

    def recognize(self, image, location):
        print("location", location) # location (195, 150, 355, 353)
        face_encodings_on_frame = face_recognition.face_encodings(image, [location])
        if face_encodings_on_frame is not None and len(face_encodings_on_frame) > 0:
            return self.recognize_person(face_encodings_on_frame[0])
        else:
            return self.recognize_person(None)