import cv2
import numpy as np
import mediapipe as mp

class YawnDetectorMAR:
    def __init__(self, mar_threshold=0.6):
        self.mar_threshold = mar_threshold
        
        # MediaPipe Face Mesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65
        )

        # Các điểm landmark môi
        self.outer_mouth = [61, 185, 40, 39, 37, 0, 267, 269, 270, 
                            409, 291, 375, 321, 405, 314, 17, 84, 
                            181, 91, 146]
        self.inner_mouth = [78, 95, 88, 178, 87, 14, 317, 402, 
                            318, 324, 308, 415, 310, 311, 312, 
                            13, 82, 81, 80, 191]

    def preprocess(self, frame):
        """Lấy landmarks mặt"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def compute_mar(self, face_landmarks, frame):
        """Tính MAR cho 1 frame"""
        if face_landmarks is None:
            return 0.0
        
        h, w, _ = frame.shape

        def euclidean_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def get_point(idx):
            return (face_landmarks.landmark[idx].x * w, 
                    face_landmarks.landmark[idx].y * h)
        
        try:
            # Các khoảng cách dọc môi
            deep_top, deep_bottom = get_point(78), get_point(95)
            center_top, center_bottom = get_point(13), get_point(14)
            left_top, left_bottom = get_point(82), get_point(87)
            right_top, right_bottom = get_point(312), get_point(317)

            inner_depth = euclidean_distance(deep_top, deep_bottom)
            v_center = euclidean_distance(center_top, center_bottom)
            v_left = euclidean_distance(left_top, left_bottom)
            v_right = euclidean_distance(right_top, right_bottom)

            left_corner, right_corner = get_point(61), get_point(291)
            mouth_width = euclidean_distance(left_corner, right_corner)
            if mouth_width < 1:
                return 0.0

            # MAR (tỉ lệ mở miệng)
            mar = (inner_depth + v_center + v_left + v_right) / (4.0 * mouth_width)
            mar *= 2.0  # Tăng độ nhạy nhẹ
            return mar
        
        except Exception as e:
            print(f"Error computing MAR: {e}")
            return 0.0

    def draw_mouth_points(self, frame, face_landmarks):
        """Vẽ các điểm môi"""
        if face_landmarks is None:
            return frame
            
        h, w, _ = frame.shape
        for idx in self.outer_mouth:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        for idx in self.inner_mouth:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame

    def predict_frame(self, frame):
        """Tính MAR cho 1 frame và gán nhãn"""
        face_landmarks = self.preprocess(frame)
        mar = self.compute_mar(face_landmarks, frame)
        frame = self.draw_mouth_points(frame, face_landmarks)
        
        label = 'yawn' if mar > self.mar_threshold else 'normal'
        
        # text = f"{label.upper()} MAR:{mar:.2f}"
        # cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
        #             1, (0, 0, 255) if label == 'yawn' else (0, 255, 0), 2)
        
        return frame, label, mar


# detector = YawnDetectorMAR()
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("End of video or cannot read the frame.")
#         break

#     frame, label, mar = detector.predict_frame(frame)
#     cv2.imshow("Yawn Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
