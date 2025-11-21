import cv2
import numpy as np
import mediapipe as mp 

class Preprocessing:
    def __init__(self, img_size=(101, 101), normalize=True, detect_both_eyes=True):
        self.img_size = img_size
        self.normalize = normalize
        self.detect_both_eyes = detect_both_eyes

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_EYE_INDICES = [
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        self.RIGHT_EYE_INDICES = [
            362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
        ]
        # -----------------------------------------------------------------

    def enhance_eye(self, eye_img):
        eye_smooth = cv2.bilateralFilter(eye_img, 7, 50, 50)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eye_equalized = clahe.apply(eye_smooth)

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

    def crop_eyes(self, frame, draw_box=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w = frame.shape[:2]  # Lấy kích thước ảnh
        eyes = []

        # ----------------------------------------------------
        # THAY ĐỔI: Dùng MediaPipe
        # 1. Chuyển sang RGB vì MediaPipe cần
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Tối ưu tốc độ
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return [], frame # Không tìm thấy khuôn mặt

        for face_landmarks in results.multi_face_landmarks:
            # 2. Lấy tất cả 478 điểm và chuyển về toạ độ pixel
            landmarks_list = []
            for lm in face_landmarks.landmark:
                # MediaPipe trả về (x, y) chuẩn hoá (0.0 -> 1.0)
                # Cần nhân với (w, h) để ra toạ độ pixel
                px, py = int(lm.x * img_w), int(lm.y * img_h)
                landmarks_list.append((px, py))
            
            landmarks = np.array(landmarks_list)

            # 3. Lặp qua các CHỈ SỐ MẮT MỚI
            for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
                
                # eye_points = landmarks[start:end + 1] # CŨ
                eye_points = landmarks[eye_indices]   # MỚI

                # --------------------------------------------------------------
                # TỪ ĐÂY TRỞ ĐI, CODE CỦA BẠN GIỮ NGUYÊN 100%
                # vì nó chỉ phụ thuộc vào `eye_points`
                # --------------------------------------------------------------
                
                x, y, w, h = cv2.boundingRect(eye_points)

                # Mở rộng box theo tỉ lệ
                margin_y = int(0.4 * h)
                margin_x = int(0.6 * w)
                cx, cy = x + w // 2, y + h // 2
                side = int(max(w + 2 * margin_x, h + 2 * margin_y))
                half = side // 2

                # Tính toạ độ crop
                x1, x2 = cx - half, cx + half
                y1, y2 = cy - half, cy + half

                # Giới hạn trong khung ảnh
                x1_pad, y1_pad = max(0, -x1), max(0, -y1)
                x2_pad, y2_pad = max(0, x2 - gray.shape[1]), max(0, y2 - gray.shape[0])

                # Cắt vùng mắt (từ ảnh xám)
                eye_crop = gray[max(0, y1):min(gray.shape[0], y2),
                               max(0, x1):min(gray.shape[1], x2)]

                # Nếu bị cắt thiếu thì pad thêm viền đen để giữ vuông
                if any([x1_pad, y1_pad, x2_pad, y2_pad]):
                    eye_crop = cv2.copyMakeBorder(
                        eye_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                        cv2.BORDER_CONSTANT, value=0
                    )

                # Resize và làm rõ
                eye_resized = cv2.resize(eye_crop, (101, 101), interpolation=cv2.INTER_CUBIC)
                eye_resized = self.enhance_eye(eye_resized)
                eyes.append(eye_resized)

                #  Vẽ khung quanh vùng mắt trên ảnh gốc
                if draw_box:
                    cv2.rectangle(frame, (max(0, x1), max(0, y1)),
                                  (min(frame.shape[1], x2), min(frame.shape[0], y2)),
                                  (0, 255, 0), 2)  # màu xanh lá
                    
                    eye_contour = eye_points.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(frame, [eye_contour], isClosed=True,
                                  color=(255, 0, 0), thickness=1)  # màu xanh dương
        return eyes, frame