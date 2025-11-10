from flask import Blueprint, render_template, Response, jsonify
import cv2
import time
import pygame
from BE.SystemState import SystemState

# Khởi tạo Blueprint
home_bp = Blueprint('home', __name__)

# Lấy instance SystemState (sẽ được inject từ app.py)
state = None

def init_home_blueprint(system_state):
    """Khởi tạo blueprint với SystemState"""
    global state
    state = system_state

def play_warning_music():
    """Phát nhạc cảnh báo"""
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_warning_music():
    """Dừng nhạc cảnh báo"""
    pygame.mixer.music.stop()

def generate_frames():
    """
    Generator function để stream frames cho web
    Thread 3: Stream frames cho web
    """
    print("📺 Stream thread started")
    last_encoded = None
    
    while True:
        try:
            # Lấy kết quả từ result queue
            if not state.result_queue.empty():
                frame, left_label, right_label, is_yawning, mar = state.result_queue.get()
                
                # Vẽ thông tin lên frame
                if state.is_drowsy:
                    cv2.putText(frame, "DROWSINESS ALERT!", (frame.shape[1]//2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Hiển thị FPS
                fps_text = f"Cam: {state.camera_manager.get_fps()} FPS | Proc: {state.processing_fps} FPS"
                cv2.putText(frame, fps_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    last_encoded = buffer.tobytes()
            
            # Nếu có frame đã encode, stream nó
            if last_encoded:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_encoded + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS stream

        except Exception as e:
            print(f"❌ Lỗi trong generate_frames: {e}")
            time.sleep(0.1)


# ==================== ROUTES ====================
@home_bp.route('/')
def app():
    return render_template('main.html')

@home_bp.route('/home')
def index():
    """Trang chủ - Dashboard giám sát"""
    return render_template('index.html')

@home_bp.route('/music')
def music_player():
    return render_template('music_player.html')

@home_bp.route('/video_feed')
def video_feed():
    """Stream video từ camera với detection overlay"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@home_bp.route('/api/state')
def get_state():
    """API lấy trạng thái hệ thống hiện tại"""
    return jsonify(state.get_state_dict())

@home_bp.route('/api/reset', methods=['POST'])
def reset():
    """API reset hệ thống về trạng thái ban đầu"""
    state.reset_system()
    return jsonify({
        'status': 'success', 
        'message': 'System reset successfully'
    })

@home_bp.route('/api/camera_status')
def camera_status():
    """API kiểm tra trạng thái camera"""
    return jsonify({
        'camera_working': state.camera_working,
        'frame_count': state.frame_count,
        'error_count': state.error_count,
        'camera_fps': state.camera_manager.get_fps(),
        'processing_fps': state.processing_fps
    })

@home_bp.route('/api/restart_camera', methods=['POST'])
def restart_camera():
    """API khởi động lại camera"""
    success = state.camera_manager.initialize_camera()
    if success:
        state.camera_manager.start()
    return jsonify({
        'status': 'success' if success else 'error',
        'message': 'Camera restarted successfully' if success else 'Failed to restart camera'
    })