import cv2
import time
import threading
import numpy as np

class CameraManager:
    """Quản lý camera với thread riêng để đọc frame liên tục"""
    def __init__(self):
        self.camera_index = 0
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.read_thread = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize_camera(self):
        """Khởi tạo camera"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                
            try:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm lag
                
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f" Camera {self.camera_index} hoạt động tốt")
                        return True
                    else:
                        print(f" Camera {self.camera_index} kết nối nhưng không đọc được frame")
                        self.cap.release()
                else:
                    print(f" Không thể mở camera {self.camera_index}")
                        
            except Exception as e:
                print(f" Lỗi khi thử camera: {e}")
                if self.cap:
                    self.cap.release()
            
            print(" Không tìm thấy camera nào hoạt động")
            return False
    
    def start(self):
        """Bắt đầu thread đọc frame"""
        self.stopped = False
        self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.read_thread.start()
        return self
    
    def _read_frames(self):
        """Thread riêng để đọc frame liên tục từ camera"""
        print(" Camera reading thread started")
        while not self.stopped:
            if self.cap and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                with self.lock:
                    self.grabbed = grabbed
                    if grabbed:
                        self.frame = frame
                        self.fps_counter += 1
                        
                        # Tính FPS
                        if time.time() - self.fps_start_time >= 1.0:
                            self.current_fps = self.fps_counter
                            self.fps_counter = 0
                            self.fps_start_time = time.time()
            else:
                time.sleep(0.1)
    
    def read(self):
        """Lấy frame mới nhất"""
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else (False, None)
    
    def get_fps(self):
        """Lấy FPS hiện tại"""
        with self.lock:
            return self.current_fps
    
    def create_test_frame(self):
        """Tạo frame test"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "CAMERA TEST MODE", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, "Camera is in test mode", (100, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (150, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame
    
    def stop(self):
        """Dừng thread đọc frame"""
        self.stopped = True
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
    
    def release(self):
        """Giải phóng camera"""
        self.stop()
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None