import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
import cv2
import os

class EyeStateModel:
    def __init__(self, input_shape=(101, 101, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Xây dựng kiến trúc CNN"""
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_dir, val_dir, batch_size=32, epochs=10):
        """Huấn luyện model với dữ liệu từ thư mục"""
        datagen = ImageDataGenerator(rescale=1./255)

        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            color_mode='grayscale',
            class_mode='binary',
            batch_size=batch_size
        )

        val_gen = datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            color_mode='grayscale',
            class_mode='binary',
            batch_size=batch_size
        )

        self.model.fit(train_gen, validation_data=val_gen, epochs=epochs)
        print("Training complete!")

    def save(self, path="eye_model.h5"):
        """Lưu model"""
        self.model.save(path)
        print(f" Model saved to {path}")

    def load(self, path="eye_model.h5"):
        """Tải model đã huấn luyện"""
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def predict(self, img):
        """Dự đoán trạng thái mắt từ ảnh bất kỳ kích thước"""
        img = np.expand_dims(img, axis=0)
        # Dự đoán
        prob = self.model.predict(img)[0, 0]
        label = "Open" if prob > 0.5 else "Closed"
        
        print(f"Dự đoán: {label} ({prob:.2f})")
        return label, prob

class Preprocessing:
    def __init__(self, img_size=(101, 101), normalize=True, detect_both_eyes=True):
        """
        Bộ tiền xử lý nâng cao cho phát hiện mắt rõ nét.
        """
        self.img_size = img_size
        self.normalize = normalize
        self.detect_both_eyes = detect_both_eyes

        # Tải model nhận diện mặt người của OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Tải model nhận diện mắt của OpenCV
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    def enhance_image(self, gray):
        """
        Cải thiện chất lượng ảnh nhưng giữ nguyên độ nét.
        """
        # Tăng độ tương phản cục bộ bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Giảm nhiễu nhưng vẫn giữ nét bằng Bilateral Filter
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        return gray

    def sharpen_eye(self, eye_img):
        """
        Làm sắc nét ảnh mắt sau khi resize.
        """
        # Unsharp masking để tăng độ nét
        gaussian = cv2.GaussianBlur(eye_img, (0, 0), 2.0)
        sharpened = cv2.addWeighted(eye_img, 1.5, gaussian, -0.5, 0)
        
        # Đảm bảo giá trị pixel trong khoảng [0, 255]
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened

    def detect_eyes(self, img):
        """
        Phát hiện mắt trong ảnh với chất lượng cao nhất.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tăng độ phân giải nếu ảnh quá nhỏ
        h, w = gray.shape
        if h < 400 or w < 400:
            scale = max(400 / h, 400 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Enhance toàn bộ ảnh trước
        gray_enhanced = self.enhance_image(gray)

        # Tìm tất cả khuôn mặt trong ảnh
        faces = self.face_cascade.detectMultiScale(
            gray_enhanced, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(101, 101)
        )
        
        eyes_crops = []

        for (x, y, w, h) in faces:
            # Tập trung vào 60% (nửa trên) khuôn mặt (nơi có mắt)
            face_roi = gray_enhanced[y:y+int(h*0.6), x:x+w]

            # Tìm mắt bên trong vùng mặt đã phát hiện
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, 
                scaleFactor=1.05,  # Giảm scale factor để phát hiện chính xác hơn
                minNeighbors=8,    # Tăng minNeighbors để giảm false positive
                minSize=(30, 30)
            )

            for (ex, ey, ew, eh) in eyes:
                # Mở rộng vùng crop để lấy đủ context
                pad_y, pad_x = int(eh * 0.25), int(ew * 0.25)
                y1 = max(0, ey - pad_y)
                y2 = min(face_roi.shape[0], ey + eh + pad_y)
                x1 = max(0, ex - pad_x)
                x2 = min(face_roi.shape[1], ex + ew + pad_x)

                eye_roi = face_roi[y1:y2, x1:x2]
                
                # Kiểm tra kích thước hợp lệ
                if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
                    continue

                # Resize với interpolation tốt nhất
                # INTER_CUBIC tốt cho downscaling
                eye_resized = cv2.resize(eye_roi, self.img_size, interpolation=cv2.INTER_CUBIC)
                
                # Làm sắc nét sau khi resize
                eye_resized = self.sharpen_eye(eye_resized)
                
                # Chuẩn hóa histogram một lần nữa - tối đa hóa độ tương phản
                eye_resized = cv2.equalizeHist(eye_resized)

                if self.normalize:
                    eye_resized = eye_resized.astype("float32") / 255.0

                # (101, 101) -> (101, 101, 1)
                eye_resized = np.expand_dims(eye_resized, axis=-1) 
                eyes_crops.append(eye_resized)

        return eyes_crops

    def preprocess_and_visualize(self, img, save_path=None):
        """
        Xử lý ảnh giống detect_eyes nhưng có thêm:
        - Trả về 2 mắt đã chuẩn hóa như trước
        - Vẽ bounding box face và eye lên ảnh gốc, trả về ảnh gốc có khung
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tăng độ phân giải nếu ảnh quá nhỏ
        h, w = gray.shape
        scale = 1.0
        if h < 400 or w < 400:
            scale = max(400 / h, 400 / w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img_disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            img_disp = img.copy()

        gray_enhanced = self.enhance_image(gray)
        faces = self.face_cascade.detectMultiScale(
            gray_enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(101, 101)
        )

        eyes_crops = []
        for (x, y, w_f, h_f) in faces:
            # Vẽ khung face
            cv2.rectangle(img_disp, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)

            face_roi = gray_enhanced[y:y+int(h_f*0.6), x:x+w_f]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(30, 30)
            )

            for (ex, ey, ew, eh) in eyes:
                pad_y, pad_x = int(eh * 0.25), int(ew * 0.25)
                y1 = max(0, ey - pad_y)
                y2 = min(face_roi.shape[0], ey + eh + pad_y)
                x1 = max(0, ex - pad_x)
                x2 = min(face_roi.shape[1], ex + ew + pad_x)
                eye_roi = face_roi[y1:y2, x1:x2]

                if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
                    continue

                # Resize và sharpen
                eye_resized = cv2.resize(eye_roi, self.img_size, interpolation=cv2.INTER_CUBIC)
                eye_resized = self.sharpen_eye(eye_resized)
                eye_resized = cv2.equalizeHist(eye_resized)
                if self.normalize:
                    eye_resized = eye_resized.astype("float32") / 255.0
                    eye_resized = np.expand_dims(eye_resized, axis=-1)

                eyes_crops.append(eye_resized)

                # Vẽ khung eye lên ảnh gốc
                ex_disp = int(x + ex - pad_x)
                ey_disp = int(y + ey - pad_y)
                ew_disp = int(ew + 2*pad_x)
                eh_disp = int(eh + 2*pad_y)
                cv2.rectangle(img_disp, (ex_disp, ey_disp), (ex_disp + ew_disp, ey_disp + eh_disp), (0, 0, 255), 2)

        # Nếu không detect được mắt -> thêm ảnh mắt trống
        if not eyes_crops:
            empty_eye = np.zeros((*self.img_size, 1), dtype=np.float32)
            eyes_crops = [empty_eye, empty_eye]
        elif len(eyes_crops) == 1: # Chỉ detect được 1 mắt -> [mắt tìm thấy, ảnh đen]
            empty_eye = np.zeros((*self.img_size, 1), dtype=np.float32)
            eyes_crops = [eyes_crops[0], empty_eye]
        else: # Nhiều hơn 2 mắt -> chỉ lấy 2 mắt đầu tiên
            eyes_crops = eyes_crops[:2]

        if save_path:
            cv2.imwrite(save_path, img_disp)

        return img_disp, eyes_crops
