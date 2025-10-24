import cv2
import numpy as np
import mediapipe as mp

class Preprocessing:
    def __init__(self, target_size=(24, 24), mode="mediapipe", save_gray=True):
        """
        Bộ tiền xử lý ảnh real-time cho phát hiện buồn ngủ.
        - target_size: kích thước ảnh đầu vào cho model
        - mode: mediapipe (chính xác cao) hoặc haar (dự phòng)
        - save_gray: có chuyển ảnh sang grayscale không
        """
        self.target_size = target_size
        self.mode = mode
        self.save_gray = save_gray

        if mode == "mediapipe":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1
            )
        else:
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def _normalize(self, img):
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32') / 255.0
        if len(img.shape) == 2:  # grayscale
            img = np.expand_dims(img, axis=-1)
        return img

    def _extract_eyes_mediapipe(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        eyes = []

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]

            # Mắt trái và phải (theo chỉ số landmark Mediapipe)
            left_eye_idx = [33, 133, 159, 145]
            right_eye_idx = [362, 263, 386, 374]

            def crop_eye(indices):
                xs = [int(landmarks.landmark[i].x * w) for i in indices]
                ys = [int(landmarks.landmark[i].y * h) for i in indices]
                x1, x2 = max(min(xs) - 5, 0), min(max(xs) + 5, w)
                y1, y2 = max(min(ys) - 5, 0), min(max(ys) + 5, h)
                return frame[y1:y2, x1:x2]

            left_eye = crop_eye(left_eye_idx)
            right_eye = crop_eye(right_eye_idx)

            if left_eye.size > 0: eyes.append(left_eye)
            if right_eye.size > 0: eyes.append(right_eye)

        return eyes

    def preprocess_frame(self, frame):
        """
        Tiền xử lý 1 frame từ camera.
        Trả về danh sách ảnh mắt đã chuẩn hóa để đưa vào model.
        """
        eyes = self._extract_eyes_mediapipe(frame) if self.mode == "mediapipe" else []

        processed = []
        for eye in eyes:
            if self.save_gray:
                eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye = self._normalize(eye)
            processed.append(eye)
        return processed
