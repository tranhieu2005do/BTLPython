import tensorflow as tf

print("Đang tải model .h5...")
# Đảm bảo đường dẫn này là đúng
model = tf.keras.models.load_model(r"D:\New folder\BTLPython\BE\model\eye_model\eye_model.h5")
print("Tải xong. Bắt đầu chuyển đổi...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Lưu model .tflite mới
with open(r"D:\New folder\BTLPython\BE\model\eye_model\eye_model.tflite", 'wb') as f:
    f.write(tflite_model)

print("✅ Đã chuyển đổi và lưu file eye_model.tflite!")