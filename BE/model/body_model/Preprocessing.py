import cv2
import mediapipe as mp
import numpy as np

class MouthPreprocessor:
    def __init__(self, resize=(128,128)):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               refine_landmarks=True)
        self.resize = resize

        # Inner lips landmarks (MediaPipe 468)
        self.inner_lips = [78,191,80,81,82,13,312,311,310,415,308,87]
        # Optional: small subset outer lips to include toàn bộ miệng
        self.outer_lips = [61,146,91,181,84,17,314,405,321,375,291,308]
        self.mouth_idx = self.inner_lips + self.outer_lips
    
    def enhance_mouth(self,mouth_img):
        """Nâng cao chất lượng ảnh miệng"""
        # Bước 1: Làm mịn nhưng giữ biên (bilateral)
        eye_smooth = cv2.bilateralFilter(mouth_img, 7, 50, 50)

        # Bước 2: Cân bằng sáng cục bộ bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        eye_equalized = clahe.apply(eye_smooth)

        # Bước 3: Làm sắc nét bằng kernel
        sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
        eye_sharp = cv2.filter2D(eye_equalized, -1, sharpen_kernel)

        # Bước 4: Hiệu chỉnh gamma để sáng đều
        gamma = 1.2
        lookUpTable = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        eye_gamma = cv2.LUT(eye_sharp, lookUpTable)

        return eye_gamma

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Lấy tất cả các điểm miệng
            points = np.array([[int(face_landmarks.landmark[i].x * w),
                                int(face_landmarks.landmark[i].y * h)] for i in self.mouth_idx])

            # Tính bounding box chuẩn: từ min/max của các điểm
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            # Padding nhỏ 5px để tránh sát quá
            x_min, y_min = max(x_min-5,0), max(y_min-5,0)
            x_max, y_max = min(x_max+5,w), min(y_max+5,h)

            # Crop và resize
            mouth_crop = frame[y_min:y_max, x_min:x_max]
            if mouth_crop.size == 0:
                return None, None
            mouth_crop_resized = cv2.resize(mouth_crop, self.resize)
            mouth_img = self.enhance_mouth(cv2.cvtColor(mouth_crop_resized, cv2.COLOR_BGR2GRAY))

            return mouth_img, (x_min, y_min, x_max, y_max)
        else:
            return None, None