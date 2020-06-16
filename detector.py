import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


class MaskDetector:
    def __init__(self, modelPath="storage/mask_detector.model"):
        self.maskNet = load_model(modelPath)

    def is_mask(self, prediction):
        (mask, withoutMask) = prediction
        return mask > withoutMask

    def predict_mask(self, faces):
        return self.maskNet.predict(faces)

    def detect_and_predict_mask(self, frame, face_locations):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []

        # loop over the detections
        for i in range(0, len(face_locations)):
            top, right, bottom, left = face_locations[i]
            print("locations", face_locations)
            face = frame[top:bottom, left:right]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)

        # only make a predictions if at least one face was detected

        preds = self.maskNet.predict(faces)

        return preds


class FaceDetector:
    def __init__(self, confidence=0.5, prototxtPath="storage/face_detector/deploy.prototxt",
                 weightsPath="storage/face_detector/res10_300x300_ssd_iter_140000.caffemodel"):
        self.confidence = confidence
        self.net = cv2.dnn.readNet(prototxtPath, weightsPath)

    def detect_faces(self, image):
        orig = image.copy()
        (height, width) = image.shape[:2]

        # construct a blob from the image
        new_width = 300
        new_height = 300
        scale_factor_x = width / new_width
        scale_factor_y = height / new_height
        blob = cv2.dnn.blobFromImage(image, 1.0, (new_width, new_height), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.confidence:
                try:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(width - 1, endX), min(height - 1, endY))
                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = image[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)
                    location = (startX, startY, endX, endY)
                    faces.append((face, location))
                except:
                    pass
        return faces
