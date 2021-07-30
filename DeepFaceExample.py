from lib.deepface.deepface import DeepFace
from lib.deepface.deepface.commons import functions
import tensorflow as tf
import json
import cv2
import time


class deepface(object):

    @staticmethod
    def checkinfo(path):
        output = []
        demography = DeepFace.analyze(path, ['age', 'gender', 'race', 'emotion'])
        output.append(demography["age"])
        output.append(demography["gender"])
        output.append(demography["dominant_race"])
        output.append(demography["dominant_emotion"])
        #print("Age: ", demography["age"])
        #print("Gender: ", demography["gender"])
        #print("Race: ", demography["dominant_race"])
        #print("Emotion: ", demography["dominant_emotion"])
        return output

    @staticmethod
    def captureStream():
        capture = cv2.VideoCapture(0)
        capture.set(3, 640)
        capture.set(4, 480)
        img_counter = 0
        frame_set = []
        start_time = time.time()
        while True:
            try:
                ret, frame = capture.read()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if time.time() - start_time >= 5: #<---- Check if 5 sec passed
                    img_name = "lib/deepface/tests/dataset/opencv_frame_{}.jpg".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    #print("{} written!".format(img_counter))
                    output = deepface.checkinfo("lib/deepface/tests/dataset/opencv_frame_{}.jpg".format(img_counter))
                    text = "Age: "+str(output[0])+"\nGender: "+str(output[1])+"\nRace:  "+str(output[2])+"\nIs: "+(str(output[3]))
                    #print(text)
                    img = cv2.imread(img_name)
                    font =  cv2.FONT_HERSHEY_SIMPLEX
                    y0, dy = 50, 50
                    for i, line in enumerate(text.split('\n')):
                        y = y0 + i*dy
                        cv2.putText(img, line, (50,y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Here is an Example', img)
                    start_time = time.time()
                img_counter += 1
            except:
                time.sleep(3)




#deepface.captureStream()
