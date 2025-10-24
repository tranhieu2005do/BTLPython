import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from preprocessing import Preprocessing 

pre = Preprocessing(target_size=(24, 24), mode="mediapipe")

img = cv2.imread(r"D:\python_code\Individual_model\BTLPYTHON\BE\download (1).jpg") 

res = pre.preprocess_frame(img)
cv2.imshow("Demo", res[1])           
cv2.waitKey(0)                   
cv2.destroyAllWindows() 