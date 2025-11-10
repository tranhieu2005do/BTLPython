import sys, os
sys.path.append(os.path.dirname(__file__))  # để tìm SystemState trong FE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
from collections import deque
import threading
from queue import Queue
from CameraManager import CameraManager
import pygame

# Audio
pygame.mixer.init()
pygame.mixer.music.load(r"D:\\python_code\\Individual_model\\BTLPYTHON\\BE\\alert.mp3")

def play_warning_music():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_warning_music():
    pygame.mixer.music.stop()
# Global state
class SystemState:
    def __init__(self):
        self.camera_manager = CameraManager()
        self.lock = threading.Lock()
        
        # Queue cho frame processing
        self.frame_queue = Queue(maxsize=1)  # Giữ tối đa 2 frames
        self.result_queue = Queue(maxsize=1)
        
        # Frame hiển thị
        self.display_frame = None
        self.frame_ready = False

        # Eye tracking
        self.left_eye_label = "Unknown"
        self.left_eye_prob = 0.0
        self.right_eye_label = "Unknown"
        self.right_eye_prob = 0.0
        self.eye_closure_start_time = None
        self.current_closure_duration = 0.0
        self.eye_closure_alert = False
        self.CONTINUOUS_CLOSURE_THRESHOLD = 4.0

        # Yawn tracking
        self.yawn_events = deque()
        self.yawn_count = 0
        self.YAWN_THRESHOLD = 6
        self.YAWN_WINDOW = 60.0
        self.is_yawning = False
        self.mar = 0.0
        self.yawn_frequency_alert = False
        self.yawn_alert_triggered = False
        self.last_yawn_detected = False

        # Music control
        self.music_playing = False
        self.music_start_time = None
        self.last_alert_time = None
        self.MUSIC_DURATION = 5.0
        self.RECOVERY_CHECK_TIME = 5.0
        self.is_drowsy = False

        # Statistics
        self.total_eye_closure_alerts = 0
        self.total_yawn_frequency_alerts = 0
        self.total_microsleeps = 0
        
        # Camera status
        self.camera_working = False
        self.frame_count = 0
        self.error_count = 0
        self.processing_fps = 0

    def get_state_dict(self):
        with self.lock:
            return {
                'eyes': {
                    'left': {'label': self.left_eye_label, 'prob': float(self.left_eye_prob)},
                    'right': {'label': self.right_eye_label, 'prob': float(self.right_eye_prob)},
                    'closure_duration': float(self.current_closure_duration),
                    'closure_threshold': float(self.CONTINUOUS_CLOSURE_THRESHOLD),
                    'alert': self.eye_closure_alert
                },
                'yawn': {
                    'is_yawning': self.is_yawning,
                    'mar': float(self.mar),
                    'count': self.yawn_count,
                    'threshold': self.YAWN_THRESHOLD,
                    'alert': self.yawn_frequency_alert
                },
                'system': {
                    'is_drowsy': self.is_drowsy,
                    'music_playing': self.music_playing,
                    'total_eye_alerts': self.total_eye_closure_alerts,
                    'total_yawn_alerts': self.total_yawn_frequency_alerts,
                    'total_microsleeps': self.total_microsleeps,
                    'camera_status': 'working' if self.camera_working else 'test_mode',
                    'camera_fps': self.camera_manager.get_fps(),
                    'processing_fps': self.processing_fps
                }
            }

    def reset_system(self):
        """Reset hệ thống"""
        with self.lock:
            self.eye_closure_start_time = None
            self.current_closure_duration = 0.0
            self.eye_closure_alert = False
            self.yawn_events.clear()
            self.yawn_frequency_alert = False
            self.yawn_alert_triggered = False
            self.last_yawn_detected = False
            if self.music_playing:
                stop_warning_music()
            self.music_playing = False
            self.music_start_time = None
            self.last_alert_time = None
            
    def release_camera(self):
        """Giải phóng camera"""
        self.camera_manager.release()