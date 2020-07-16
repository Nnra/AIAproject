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
import operator
from collections import Counter
from imutils.face_utils import FaceAligner
from tensorflow.keras.models import load_model, model_from_json
from function import *

class FaceEmotion:
    def __init__(self, input='video', source=None, output_dir='output', model='models/base_emotion_classification_model', conf_threshold=0.8, fps=25, show=False):
        self.input=input
        self.output_dir=output_dir
        self.model=model
        self.conf_threshold=conf_threshold
        self.fps=fps
        self.show=show
        self.source=source
        
    def predict(self):
        print('start predict')
        
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

        # ___ EMOTION RECOGNITION MODEL ___
        emotion_detector = None

        if os.path.isfile("{}.json".format(self.model)):
            json_file = open("{}.json".format(self.model), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            emotion_detector = model_from_json(loaded_model_json)
            emotion_detector.load_weights("{}.h5".format(self.model))

        elif os.path.isfile("{}.hdf5".format(self.model)):
            emotion_detector = load_model("{}.hdf5".format(self.model), compile=False)            
        else:
            print("Error. File {} was not found!".format(self.model))
            sys.exit(-1) 

        frames_resolution = [emotion_detector.input_shape[-3], emotion_detector.input_shape[-2], emotion_detector.input_shape[-1]]

        # ___ FACE ALIGNER ___ (uses emotion recognition model input shape)
        predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        fa = FaceAligner(predictor, desiredFaceWidth=emotion_detector.input_shape[-3])

        # output file path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        name = "result" + (os.path.normpath(self.source).split("\\")[-1] if (self.input != "camera") else ".avi")
        save_path = os.path.join(self.output_dir, name)
        print(f'save_path: {save_path}')

        fourcc = cv2.VideoWriter_fourcc(*'VP80') # initral >> MPEG # vp80 >> webm # mp4 >> mp4v
        vid = None

        emotion_list =[]
        total = []
        total_emotion = []
        cam = cv2.VideoCapture(self.source)
        _, image = cam.read()
        print("Camera image shape: {}x{}".format(image.shape[1], image.shape[0]))
        get_emotion = ''
        max_percent = ''
        while(cam.isOpened()):
            # read image
            _, image = cam.read()
            if _:
                # detect emotions
                success, coords, emotion_dict = detect_emotions(image, emotion_detector, face_detector, self)
                if(success) and coords!=[] and emotion_dict!={}:
                    total.append(emotion_dict)
                    total_emotion.append(emotion_dict['emotion'])
                    if len(emotion_list)<=7:
                        emotion_list.append(emotion_dict)
                    else:
                        emotion_list.pop(0)
                        emotion_list.append(emotion_dict)
                        max_percent = sorted(emotion_list, key=lambda k: k['emotion_percent'], reverse=True)[0]['emotion_percent']
                        get_emotion = sorted(emotion_list, key=lambda k: k['emotion_percent'], reverse=True)[0]['emotion']

                    print('=====================')
                    #print(total)
                    # 對人臉畫出方框
                    cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
                    if max_percent!='':
                        cv2.putText(image, str(max_percent) + "%", (coords[0], coords[3] + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if get_emotion !='':
                        cv2.putText(image, get_emotion, (coords[0], coords[3] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # show result
#                     cv2.imshow('image', image)
                    fps_time = time.time()

                    if vid == None:
                        print(f'vid is None: save_path= {save_path}')
                        vid = cv2.VideoWriter(filename=save_path, fourcc=fourcc, fps=float(self.fps), apiPreference=cv2.CAP_FFMPEG, frameSize=(image.shape[1],image.shape[0]))

                    # save output
                    print('save output')
                    vid.write(image)

                    # return 所有情緒的總和,最大值
                    #　tracking : 偵測人臉

                else:
                    print("Unsuccessfull image processing")

                # wait Esc to be pushed
                if cv2.waitKey(1) == 27:
                    break
            else:
                break

        vid.release()
        cv2.destroyAllWindows()
        print('end predict')
        new_total = sorted(total, key=lambda k: k['emotion'], reverse=True) 
        new_total = sorted(new_total, key=lambda k: k['emotion_percent'], reverse=True) 
        count = Counter(total_emotion)
        get_max_count= count.most_common(1)[0][0]
        return get_max_count,new_total