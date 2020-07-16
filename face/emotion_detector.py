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
from imutils.face_utils import FaceAligner
from tensorflow.keras.models import load_model, model_from_json
from function import *

#     N   EN
#     0 = neutral
#     1 = calm
#     2 = happy
#     3 = sad
#     4 = angry
#     5 = fearful
#     6 = disgust
#     7 = surprised

emotions_en = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

parser = argparse.ArgumentParser(description='Emotion recognition')
parser.add_argument('--camera_number', type=int, default=0)
parser.add_argument('--input', type=str, default='camera', help='can be "image", "camera" or "video"')
parser.add_argument('--source', type=str, default="images/i.jpg")
parser.add_argument('--output_dir', type=str, default="output", help='directory to save result image or video')
parser.add_argument('--model', type=str, default="models/base_emotion_classification_model", help='emotion detector model')
parser.add_argument('--conf_threshold', type=float, default=0.8, help='face detector threshold')
parser.add_argument('--fps', type=int, default=25, help='output video frame rate (for video input only)')
parser.add_argument('--show', dest='show', action='store_false')
parser.set_defaults(show=True)
args = parser.parse_args()

# X server error is server not have GUI
if __name__ == '__main__':
    # ___ FACE DETECTOR MODEL ___
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    face_detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # ___ EMOTION RECOGNITION MODEL ___
    emotion_detector = None

    if os.path.isfile("{}.json".format(args.model)):
        json_file = open("{}.json".format(args.model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_detector = model_from_json(loaded_model_json)
        emotion_detector.load_weights("{}.h5".format(args.model))

    elif os.path.isfile("{}.hdf5".format(args.model)):
        emotion_detector = load_model("{}.hdf5".format(args.model), compile=False)            
    else:
        print("Error. File {} was not found!".format(args.model))
        sys.exit(-1) 

    frames_resolution = [emotion_detector.input_shape[-3], emotion_detector.input_shape[-2], emotion_detector.input_shape[-1]]

    # ___ FACE ALIGNER ___ (uses emotion recognition model input shape)
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=emotion_detector.input_shape[-3])

    # output file path
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    name = "result" + (os.path.normpath(args.source).split("\\")[-1] if (args.input != "camera") else ".avi")
    save_path = os.path.join(args.output_dir, name)
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80') # initral >> MPEG # vp80 >> webm # mp4 >> mp4v
    vid = None

    percent_dict ={}
    emotion_list =[]
    cam = cv2.VideoCapture(args.source)
    _, image = cam.read()
    print("Camera image shape: {}x{}".format(image.shape[1], image.shape[0]))
    get_emotion = ''
    max_percent = ''
    while(cam.isOpened()):
        # read image
        _, image = cam.read()
        if _:
            # detect emotions
            success, coords, emotion_dict = detect_emotions(image, emotion_detector, face_detector, args)
            if(success) and coords!=[] and emotion_dict!={}:
                if len(percent_dict)<=7:
                    percent_dict.update({str(len(percent_dict)):emotion_dict['emotion_percent']})
                    emotion_list.append(emotion_dict['emotion'])
                else:
                    max_percent = max(percent_dict.items(), key=operator.itemgetter(1))[1]
                    get_emotion = emotion_list[int(max(percent_dict.items(), key=operator.itemgetter(1))[0])]
                    print('=====================')
                    print(emotion_list)
                    print(percent_dict)
                    percent_dict = {}
                    emotion_list=[]
                # 對人臉畫出方框
                cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 2)
                if max_percent!='':
                    cv2.putText(image, str(max_percent) + "%", (coords[0], coords[3] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if get_emotion !='':
                    cv2.putText(image, get_emotion, (coords[0], coords[3] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # show result
                #cv2.imshow('image', image)
                fps_time = time.time()

                if vid == None:
                    vid = cv2.VideoWriter(filename=save_path, fourcc=fourcc, fps=float(args.fps), apiPreference=cv2.CAP_FFMPEG, frameSize=(image.shape[1],image.shape[0]))

                # save output
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
    
    
