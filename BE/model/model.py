import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
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
