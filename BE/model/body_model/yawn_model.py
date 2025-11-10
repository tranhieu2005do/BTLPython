import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class YawnDetectionModel:
    def __init__(self, input_size=(128,128,3)):
        self.input_size = input_size
        self.model = self.build_model()


    def build_model(self):  
        """Xây dựng CNN đơn giản cho Yawn Detection"""  
        model = Sequential([  
            Conv2D(32,(3,3),activation='relu',input_shape=self.input_size),  
            MaxPooling2D(2,2),  
            Conv2D(64,(3,3),activation='relu'),  
            MaxPooling2D(2,2),  
            Conv2D(128,(3,3),activation='relu'),  
            MaxPooling2D(2,2),  
            Flatten(),  
            Dense(128,activation='relu'),  
            Dropout(0.5),  
            Dense(1,activation='sigmoid')  
        ])  
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
        return model  

    def train_model(self, dataset_dir, batch_size=32, epochs=20, validation_split=0.2):  
        """Huấn luyện model từ dataset crop sẵn"""  
        datagen = ImageDataGenerator(rescale=1./255,  
                                    rotation_range=15,  
                                    width_shift_range=0.1,  
                                    height_shift_range=0.1,  
                                    horizontal_flip=True,  
                                    validation_split=validation_split)  

        train_gen = datagen.flow_from_directory(  
            dataset_dir,  
            target_size=self.input_size[:2],  
            batch_size=batch_size,  
            class_mode='binary',  
            subset='training'  
        )  

        val_gen = datagen.flow_from_directory(  
            dataset_dir,  
            target_size=self.input_size[:2],  
            batch_size=batch_size,  
            class_mode='binary',  
            subset='validation'  
        )  

        self.model.fit(train_gen, validation_data=val_gen, epochs=epochs)  

    def predict_frame(self, frame_crop):  
        """Dự đoán một frame đã crop"""  
        img = frame_crop.astype('float32')/255.0  
        img = np.expand_dims(img, axis=0)  
        pred = self.model.predict(img, verbose=0)[0][0]  
        label = "Yawning" if pred>0.5 else "Normal"  
        return label, float(pred)  

    def save_model(self, path):  
        """Lưu toàn bộ model (architecture + weights)"""  
        self.model.save(path)  
        print(f"Model đã được lưu tại {path}")  

    def load_model(self, path):  
        """Load toàn bộ model đã lưu"""  
        self.model = load_model(path)  
        print(f"Model đã được load từ {path}")  
    
