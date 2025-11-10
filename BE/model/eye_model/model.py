import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    GlobalAveragePooling2D, Reshape, multiply, Input, BatchNormalization, Activation
)
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os


def se_block(input_tensor, reduction=16):
    """Squeeze-and-Excitation Block"""
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // reduction, activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    return multiply([input_tensor, se])


class EyeStateModel:
    def __init__(self, input_shape=(101, 101, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """Xây dựng kiến trúc CNN + SE Blocks"""
        inputs = Input(shape=self.input_shape)

        # --- Block 1 ---
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = se_block(x)

        # --- Block 2 ---
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = se_block(x)

        # --- Block 3 ---
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)
        x = se_block(x)

        # --- Classifier ---
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_dir, val_dir, batch_size=32, epochs=10):
        """Huấn luyện model với dữ liệu từ thư mục"""
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            color_mode='grayscale',
            class_mode='binary',
            batch_size=batch_size
        )

        val_gen = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            color_mode='grayscale',
            class_mode='binary',
            batch_size=batch_size
        )

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs
        )
        print("✅ Training complete!")

    def save(self, path="eye_model.h5"):
        """Lưu model"""
        self.model.save(path)
        print(f"💾 Model saved to {path}")

    def load(self, path="eye_model.h5"):
        """Tải model đã huấn luyện"""
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={"se_block": se_block}
        )
        print(f"✅ Model loaded from {path}")

    def predict(self, img):
        """Dự đoán trạng thái mắt từ ảnh bất kỳ kích thước"""
        img = np.expand_dims(img, axis=0)
        prob = self.model.predict(img, verbose=0)[0, 0]
        label = "Open" if prob > 0.5 else "Closed"
        print(f"Dự đoán: {label} ({prob:.2f})")
        return label, prob
