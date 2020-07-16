import cv2
import argparse
import os
import time
import json
import sys
import dlib
import pandas as pd
import numpy as np
import imutils
from imutils.face_utils import FaceAligner
from tensorflow.keras.models import load_model, model_from_json

emotions_en = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
emotions_ru = ["нейтральное состояние", "спокойствие", "радость", "грусть", "злость", "испуг", "отвращение", "удивление"]

def get_faces(image, face_detector):
    ''' Get faces information from image
    Args:
        image (array): Image to process
        face_detector: loaded model for face detection
    Returns:
        bool: True if at least 1 face was found
        array of vectors: one vector for each face (array of [0,0,confidence,x1,y1,x2,y2])
    '''

    success = True
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    face_detector.setInput(blob)
    faces = face_detector.forward()

    if(faces.shape[2] == 0):
        print("No faces were found")
        success = False
    return success, faces


def detect_emotion_image(face, emotion_detector):
    ''' Get emotion prediction from image
    Args:
        face (array): Face picture to process
        emotion_detector: loaded model for face expression classification
    Returns:
        vector: confidence for each emotion
    '''

    emotion_predictions = emotion_detector.predict(face)[0]
    return emotion_predictions


def detect_emotion_video(faces, emotion_detector):
    ''' Get emotion prediction from a number of frames
    Args:
        faces (array of arrays): Array of face pictures to process
        emotion_detector: loaded model for face expression classification
    Returns:
        vector: confidence for each emotion
    '''

    faces = np.asarray(faces)
    emotion_predictions = emotion_detector.predict(faces)[0]
    return emotion_predictions

def detect_emotions(image, emotion_detector, face_detector, args):
    ''' Detects emotion on image
    Args:
        image (array): image to detect emotions on
        emotion_detector: loaded model for face expression classification
        face_detector: loaded model for face detection
        args: command arguments
    Returns:
        bool: True if process was successfull
        image (array): input image after processing
    '''
    global images

    # detect faces
    success, faces = get_faces(image, face_detector)
    coords = []
    emotion_dict = {}
    if (success):
        # loop through all found faces
        for f in range(faces.shape[2]):
            confidence = faces[0, 0, f, 2]
            if confidence > args.conf_threshold:
                x1 = int(faces[0, 0, f, 3] * image.shape[1])
                y1 = int(faces[0, 0, f, 4] * image.shape[0])
                x2 = int(faces[0, 0, f, 5] * image.shape[1])
                y2 = int(faces[0, 0, f, 6] * image.shape[0])
                coords = [x1, y1, x2, y2]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # detected_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                detected_face = image[y1:y2, x1:x2]
                if detected_face.size != 0:

                    # resize, normalize and save the frame (convert to grayscale if frames_resolution[-1] == 1)
                    if (emotion_detector.input_shape[-1] == 1):
                        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                    detected_face = cv2.resize(detected_face,
                                               (emotion_detector.input_shape[-3], emotion_detector.input_shape[-2]))
                    detected_face = cv2.normalize(detected_face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                  dtype=cv2.CV_32F)

                    # ___ one image emotion detector
                    if len(emotion_detector.input_shape) == 4:
                        detected_face = np.expand_dims(detected_face, axis=0)

                        # if emotion detector input is grayscale image
                        if emotion_detector.input_shape[-1] == 1:
                            detected_face = np.expand_dims(detected_face, axis=3)
                        emotion_predictions = detect_emotion_image(detected_face, emotion_detector)
                        emotion_probability = np.max(emotion_predictions)  # >> xx %
                        emotion_percent = np.round(emotion_probability * 100)
                        emotion_label_arg = np.argmax(emotion_predictions)  # >> 情緒標籤
                        emotion_en = emotions_en[emotion_label_arg]  # 情緒種類
                        emotion_dict = {'emotion': emotion_en, 'emotion_percent': emotion_percent}
                    elif args.input == "image":
                        print("This emotion detector does not support emotion classification on 1 image")
                        success = False
                        return success, image

                    # ___ multiple image emotion detector
                    if len(emotion_detector.input_shape) == 5:
                        if emotion_detector.input_shape[-1] == 1:
                            detected_face = np.expand_dims(detected_face, axis=4)
                        images.append(detected_face)

                        # if enough images in storage
                        if emotion_detector.input_shape[1] == len(images):

                            images_arr = np.expand_dims(np.asarray(images), axis=0)
                            emotion_predictions = detect_emotion_video(images_arr, emotion_detector)
                            emotion_probability = np.max(emotion_predictions) # >> xx %
                            emotion_percent = np.round(emotion_probability * 100)
                            emotion_label_arg = np.argmax(emotion_predictions) # >> 情緒標籤
                            emotion_en = emotions_en[emotion_label_arg] # 情緒種類
                            emotion_dict = {'emotion': emotion_en, 'emotion_percent': emotion_percent}

    else:
        print("Unsuccessfull image processing")
        success = False
    return success, coords, emotion_dict
