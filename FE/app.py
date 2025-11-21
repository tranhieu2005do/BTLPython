import sys, os
sys.path.append(os.path.dirname(__file__))  # để tìm SystemState trong FE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template
import time
from collections import deque
import threading
import pygame
import cv2
import atexit
import importlib
import BE.music_routes
importlib.reload(BE.music_routes)
# Import SystemState và Blueprints
from BE.SystemState import SystemState
from BE.music_routes import music_bp
from BE.home import home_bp, init_home_blueprint
from BE.music_routes import music_bp

# Khởi tạo Flask app
app = Flask(__name__)

# Cấu hình upload
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.register_blueprint(music_bp)
# Import models
from BE.model.eye_model.model import EyeStateModel
from BE.model.eye_model.preprocessing import Preprocessing
from BE.model.body_model.model import YawnDetectorMAR

# Initialize models
print("Initializing AI models...")
detector = YawnDetectorMAR()
pre = Preprocessing(img_size=(101, 101))
model = EyeStateModel()
model.load(r"D:\New folder\BTLPython\BE\model\eye_model\eye_model.h5")
print("Models loaded successfully")

# Initialize Audio
pygame.mixer.init()
pygame.mixer.music.load(r"D:\New folder\BTLPython\BE\alert.mp3")
print("🔊 Audio system initialized")

def play_warning_music():
    """Phát nhạc cảnh báo"""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_warning_music():
    """Dừng nhạc cảnh báo"""
    pygame.mixer.music.stop()

# Global SystemState
state = SystemState()

# Inject SystemState vào home blueprint
init_home_blueprint(state)

# Đăng ký Blueprints
app.register_blueprint(home_bp)
# app.register_blueprint(music_bp)


# ==================== THREADS ====================

def frame_capture_thread():
    """
    Thread 1: Đọc frame từ camera và đưa vào queue
    Chạy liên tục để capture frames từ camera
    """
    print("🎥 Frame capture thread started")
    while True:
        try:
            grabbed, frame = state.camera_manager.read()
            
            if grabbed and frame is not None:
                # Nếu queue đầy, bỏ qua frame cũ
                if not state.frame_queue.full():
                    state.frame_queue.put((time.time(), frame))
                state.camera_working = True
            else:
                state.camera_working = False
                time.sleep(0.01)
                
        except Exception as e:
            print(f"❌ Lỗi trong frame_capture_thread: {e}")
            time.sleep(0.1)


def detection_thread():
    """
    Thread 2: Xử lý detection từ queue
    Phân tích eye closure và yawn detection theo custom rules
    """
    while True:
        try:
            # Lấy frame từ queue
            if state.frame_queue.empty():
                time.sleep(0.001)
                continue
                
            # Lấy frame MỚI NHẤT
            current_time, frame = state.frame_queue.get() 
            
            # Vứt bỏ TẤT CẢ các frame cũ đã bị ứ đọng
            while not state.frame_queue.empty():
                # Lấy và bỏ qua
                current_time, frame = state.frame_queue.get()
            
            # === DỰ ĐOÁN NGÁP ===
            _, yawn_label, mar = detector.predict_frame(frame)
            is_yawning = (yawn_label == "yawn")

            # === DỰ ĐOÁN MẮT ===
            eyes, _ = pre.crop_eyes(frame)
            
            both_eyes_closed = False
            label1, label2 = "Unknown", "Unknown"
            prob1, prob2 = 0.0, 0.0

            if len(eyes) >= 2:
                try:
                    label1, prob1 = model.predict(eyes[0])
                    label2, prob2 = model.predict(eyes[1])
                    both_eyes_closed = (label1 == "Closed" and label2 == "Closed")
                except Exception as e:
                    print(f"❌ Lỗi khi predict mắt: {e}")
            else:
                both_eyes_closed = False
                label1, label2 = "NoEyes", "NoEyes"

            # ===================================================================
            # RULE 1: THEO DÕI THỜI GIAN NHẮM MẮT LIÊN TỤC
            # ===================================================================
            if both_eyes_closed:
                if state.eye_closure_start_time is None:
                    state.eye_closure_start_time = current_time
                    # print(f"👁 Eyes closed at {time.strftime('%H:%M:%S')}")
                
                state.current_closure_duration = current_time - state.eye_closure_start_time
                
                if state.current_closure_duration >= state.CONTINUOUS_CLOSURE_THRESHOLD:
                    if not state.eye_closure_alert:
                        state.eye_closure_alert = True
                        state.total_eye_closure_alerts += 1
                        state.total_microsleeps += 1
                        print(f"⚠️ EYE CLOSURE ALERT #{state.total_eye_closure_alerts}: Eyes closed for {state.current_closure_duration:.1f}s!")
            else:
                if state.eye_closure_start_time is not None:
                    duration = current_time - state.eye_closure_start_time
                    if duration < state.CONTINUOUS_CLOSURE_THRESHOLD:
                        print(f"👁 Eyes opened after {duration:.1f}s (< {state.CONTINUOUS_CLOSURE_THRESHOLD}s threshold)")
                
                state.eye_closure_start_time = None
                state.current_closure_duration = 0.0
                state.eye_closure_alert = False

            # ===================================================================
            # RULE 2: THEO DÕI TẦN SUẤT NGÁP
            # ===================================================================
            if is_yawning and not state.last_yawn_detected:
                state.yawn_events.append(current_time)
                print(f"🥱 Yawn detected at {time.strftime('%H:%M:%S')} (Total in buffer: {len(state.yawn_events)})")
            
            state.last_yawn_detected = is_yawning

            # Xóa các yawn events cũ hơn 60s
            cutoff_time = current_time - state.YAWN_WINDOW
            while state.yawn_events and state.yawn_events[0] < cutoff_time:
                state.yawn_events.popleft()

            state.yawn_count = len(state.yawn_events)
            if state.yawn_count >= state.YAWN_THRESHOLD:
                if not state.yawn_frequency_alert:
                    state.yawn_frequency_alert = True
                    state.yawn_alert_triggered = False
                    state.total_yawn_frequency_alerts += 1
                    print(f"⚠️ YAWN FREQUENCY ALERT #{state.total_yawn_frequency_alerts}: {state.yawn_count} yawns in last 60s!")
            else:
                state.yawn_frequency_alert = False
                if state.yawn_alert_triggered:
                    state.yawn_alert_triggered = False
                    print(f"✅ Yawn frequency normalized (< {state.YAWN_THRESHOLD} yawns)")

            # ===================================================================
            # LOGIC CẢNH BÁO VÀ ÂM THANH
            # ===================================================================
            should_play_music = False
            state.is_drowsy = False
            
            if state.eye_closure_alert:
                should_play_music = True
                state.is_drowsy = True
            elif state.yawn_frequency_alert and not state.yawn_alert_triggered:
                should_play_music = True
                state.yawn_alert_triggered = True
                state.is_drowsy = True

            if should_play_music:
                if not state.music_playing:
                    play_warning_music()
                    state.music_playing = True
                    state.music_start_time = current_time
                    
                    alert_reasons = []
                    if state.eye_closure_alert:
                        alert_reasons.append(f"Eyes closed {state.current_closure_duration:.1f}s")
                    if state.yawn_frequency_alert:
                        alert_reasons.append(f"{state.yawn_count} yawns/min (1st alert)")
                    
                    print(f"🚨 ALERT STARTED: {' | '.join(alert_reasons)}")
                
                state.last_alert_time = current_time

            # Tự động tắt nhạc sau khi hồi phục
            if state.music_playing:
                time_since_start = current_time - state.music_start_time
                time_since_alert = current_time - state.last_alert_time if state.last_alert_time else 0
                
                if time_since_start >= state.MUSIC_DURATION and time_since_alert >= state.RECOVERY_CHECK_TIME:
                    stop_warning_music()
                    state.music_playing = False
                    print(f"🔇 Music stopped (played {time_since_start:.1f}s, recovered for {time_since_alert:.1f}s)")
                    
                    print("🔄 Auto-reset after alert cleared...")
                    state.eye_closure_start_time = None
                    state.current_closure_duration = 0.0
                    state.eye_closure_alert = False
                    state.yawn_events.clear()
                    state.yawn_frequency_alert = False
                    state.yawn_alert_triggered = False
                    state.last_yawn_detected = False
                    state.music_start_time = None
                    state.last_alert_time = None
                    print("✅ System reset complete!")

            # Cập nhật state
            with state.lock:
                state.left_eye_label = label1
                state.left_eye_prob = prob1
                state.right_eye_label = label2
                state.right_eye_prob = prob2
                state.is_yawning = is_yawning
                state.mar = mar
                state.frame_count += 1

            # Đưa kết quả vào result queue để stream
            if not state.result_queue.full():
                state.result_queue.put((frame, label1, label2, is_yawning, mar))
            
            # Tính FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                state.processing_fps = fps_counter
                fps_counter = 0
                fps_start = time.time()
            
        except Exception as e:
            print(f"❌ Lỗi trong detection_thread: {e}")
            time.sleep(0.01)


def cleanup():
    """Dọn dẹp khi ứng dụng kết thúc"""
    print("🧹 Đang dọn dẹp tài nguyên...")
    state.release_camera()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("✅ Cleanup complete!")

atexit.register(cleanup)



# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 DROWSINESS DETECTION SYSTEM")
    print("=" * 60)
    
    # Khởi tạo và start camera
    if state.camera_manager.initialize_camera():
        state.camera_manager.start()
        print("✅ Camera initialized successfully")
    else:
        print("❌ Failed to initialize camera")
    
    # Khởi chạy 3 threads
    capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
    detect_thread = threading.Thread(target=detection_thread, daemon=True)
    
    capture_thread.start()
    detect_thread.start()
    
    print("=" * 60)
    print("THREAD ARCHITECTURE:")
    print("Thread 1: Camera capture (continuous reading)")
    print("Thread 2: AI detection (eye + yawn)")
    print("Thread 3: Web streaming (Flask)")
    print("=" * 60)
    print("🌐 Server URLs:")
    print("   Home page:      http://localhost:5000/home")
    print("   Music player:   http://localhost:5000/music")
    print("   Camera status:  http://localhost:5000/api/camera_status")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

