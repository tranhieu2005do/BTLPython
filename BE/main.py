import importlib
import preprocessing
importlib.reload(preprocessing)
from model.model import EyeStateModel
from preprocessing import Preprocessing
import cv2

pre = Preprocessing(img_size=(101, 101))
model = EyeStateModel()

model.load("eye_model.h5")
cap = cv2.VideoCapture(0)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được camera")
        break
    img, eyes = pre.preprocess_and_visualize(frame)

    # Hiển thị frame
    cv2.imshow("Camera", img)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
