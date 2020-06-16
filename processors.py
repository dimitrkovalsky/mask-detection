from abc import abstractmethod

import cv2
import face_recognition

from detector import MaskDetector, FaceDetector
from recognizer import Recognizer


class Config:
    def __init__(self):
        self.drawLandmarks = True
        self.enableFaceRecognition = True
        self.enableFaceBlurring = False
        self.blurredFaces = []


class BaseProcessor:
    def __init__(self):
        self.config = Config()

    @abstractmethod
    def get_face_locations(self, image):
        pass

    @abstractmethod
    def process_detected_faces(self, image, locations):
        pass

    def get_rectangle_title(self, in_mask, person_name):
        mask_title = ""
        if in_mask:
            mask_title = "In Mask"
        else:
            mask_title = "No Mask"
        if person_name is None:
            return mask_title
        else:
            return person_name + " " + mask_title

    def draw_face_rectangle(self, image, location, title, color):
        top, right, bottom, left = location
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        y = top - 15 if top - 15 > 15 else top + 15
        image = cv2.putText(image, title, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)
        return image

    def draw_landmarks(self, face_landmarks, image):
        for name, list_of_points in face_landmarks.items():
            for landmark_num, xy in enumerate(list_of_points, start=1):
                cv2.circle(image, (xy[0], xy[1]), 2, (0, 125, 125), -1)

    def run(self):
        print("CV version", cv2.__version__)
        video_capture = cv2.VideoCapture(0)
        # print(video_capture)
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            image = frame

            locations = self.get_face_locations(image)
            if locations is not None and len(locations) > 0:
                self.process_detected_faces(image, locations)

            # print(type(draw))
            imshow = cv2.imshow('Video', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def process_single_image(self, image):
        locations = self.get_face_locations(image)
        if len(locations) > 0:
            self.process_detected_faces(image, locations)

    def run_single_image(self, image_path):
        image = cv2.imread(image_path)
        locations = self.get_face_locations(image)
        if len(locations) > 0:
            self.process_detected_faces(image, locations)

        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class DlibProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.recognizer = Recognizer()
        self.detector = MaskDetector()

    def process_detected_faces(self, image, locations):
        self.process_faces(image, locations)

    def get_face_locations(self, image):
        return face_recognition.face_locations(image)

    def process_faces(self, image, locations):
        mask_per_person = self.detector.detect_and_predict_mask(image, locations)
        face_encodings_on_frame = face_recognition.face_encodings(image, locations)
        face_landmarks_list = face_recognition.face_landmarks(image, locations)

        number_of_faces = len(face_landmarks_list)
        number_of_face_encoded = len(face_encodings_on_frame)
        print("I found {} face(s) and encoded {} faces in this photograph.".format(number_of_faces,
                                                                                   number_of_face_encoded))
        if len(mask_per_person) != len(locations):
            return

        for index in range(0, len(locations)):

            location = locations[index]
            in_mask = self.detector.is_mask(mask_per_person[index])
            print("In mask", in_mask)

            if in_mask:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            if self.config.drawLandmarks:
                face_landmarks = face_landmarks_list[index]
                self.draw_landmarks(face_landmarks, image)

            if self.config.enableFaceRecognition:
                print("location dlib ", location)
                face_encoding = face_encodings_on_frame[index]
                person_name = self.recognizer.recognize_person(face_encoding)
                print("Recognized : ", person_name)
            else:
                person_name = None

            image = self.draw_face_rectangle(image, location, self.get_rectangle_title(in_mask, person_name), color)


class CustomModelProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.recognizer = Recognizer()
        self.detector = MaskDetector()
        self.face_detector = FaceDetector()
        self.dlibProcessor = DlibProcessor()

    def get_face_locations(self, image):
        return self.face_detector.detect_faces(image)

    def process_detected_faces(self, image, faces):
        if faces is None or len(faces) == 0:
            return

        detected_faces, locations = [], []
        for x, y in faces:
            detected_faces.append(x)
            locations.append(y)

        mask_per_person = self.detector.predict_mask(detected_faces)
        if len(mask_per_person) < len(faces):
            return
        all_without_mask = self.is_all_without_masks(mask_per_person)
        if all_without_mask:
            self.dlibProcessor.process_single_image(image)
        else:
            self.process_with_masks(faces, image, locations, mask_per_person)

    def process_with_masks(self, faces, image, locations, mask_per_person):
        for index in range(0, len(faces)):
            location = locations[index]
            (mask, withoutMask) = mask_per_person[index]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            in_mask = mask > withoutMask
            label = "Mask" if in_mask else "No Mask"
            if label == "Mask":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            image = self.draw_face_rectangle(image, location, label, color)

    def is_all_without_masks(self, mask_per_persons):
        if mask_per_persons is None or len(mask_per_persons) == 0:
            return True

        for index in range(0, len(mask_per_persons)):
            (mask, without_mask) = mask_per_persons[index]
            if mask > without_mask:
                return False
        return True
