from flask import Blueprint, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime


# Tạo Blueprint cho music module
music_bp = Blueprint('music', __name__)

# Cấu hình thư mục
UPLOAD_FOLDER = r'D:\\python_code\\Individual_model\\BTLPYTHON\\FE\\statics\\music'
COVER_FOLDER = r'D:\\python_code\\Individual_model\\BTLPYTHON\\FE\\statics\\covers'
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024   # 5MB

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COVER_FOLDER, exist_ok=True)

# File lưu metadata của bài hát
SONGS_DB = os.path.join(os.path.dirname(__file__), 'songs_database.json')


def allowed_file(filename, allowed_extensions):
    """Kiểm tra file extension có hợp lệ không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def load_songs_db():
    """Load danh sách bài hát từ file JSON"""
    if os.path.exists(SONGS_DB):
        with open(SONGS_DB, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_songs_db(songs):
    """Lưu danh sách bài hát vào file JSON"""
    with open(SONGS_DB, 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)

@music_bp.route('/api/music/list', methods=['GET'])
def list_songs():
    """Lấy danh sách tất cả bài hát"""
    try:
        songs = load_songs_db()
        return jsonify({
            'status': 'success',
            'songs': songs,
            'total': len(songs)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@music_bp.route('/api/music/upload', methods=['POST'])
def upload_song():
    """Upload bài hát mới"""
    try:
        # Kiểm tra request
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        title = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()
        
        if not title or not artist:
            return jsonify({'status': 'error', 'message': 'Title and artist are required'}), 400
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No audio file selected'}), 400
        
        # Kiểm tra extension
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'status': 'error', 'message': 'Invalid audio file format. Allowed: mp3, wav, ogg'}), 400

        # Kiểm tra kích thước
        audio_file.seek(0, os.SEEK_END)
        audio_size = audio_file.tell()
        audio_file.seek(0)
        if audio_size > MAX_AUDIO_SIZE:
            return jsonify({'status': 'error', 'message': f'Audio file too large. Max size: {MAX_AUDIO_SIZE / (1024*1024)}MB'}), 400
        
        # --- Lưu file audio ---
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        audio_file.save(audio_path)
        
        # --- Lưu cover image ---
        cover_filename = 'default.jpg'
        if 'cover' in request.files:
            cover_file = request.files['cover']
            if cover_file.filename != '' and allowed_file(cover_file.filename, ALLOWED_IMAGE_EXTENSIONS):
                cover_file.seek(0, os.SEEK_END)
                cover_size = cover_file.tell()
                cover_file.seek(0)
                if cover_size <= MAX_IMAGE_SIZE:
                    cover_filename = secure_filename(cover_file.filename)
                    cover_path = os.path.join(COVER_FOLDER, cover_filename)
                    cover_file.save(cover_path)
                    print(f"✅ Cover saved to: {cover_path}")
        
        audio_url = url_for('static', filename=f'music/{audio_filename}')
        cover_url = url_for('static', filename=f'covers/{cover_filename}')

        # --- Tạo metadata cho database ---
        songs = load_songs_db()
        new_id = max([s['id'] for s in songs], default=0) + 1

        new_song = {
            'id': new_id,
            'title': title,
            'artist': artist,
            'filename': audio_url,   
            'cover': cover_url,      
            'upload_date': datetime.now().isoformat(),
            'size': audio_size
        }

        songs.append(new_song)
        save_songs_db(songs)
        
        return jsonify({
            'status': 'success', 
            'message': 'Song uploaded successfully', 
            'song': new_song
        })

    except Exception as e:
        print(f" Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Upload failed: {str(e)}'}), 500

@music_bp.route('/api/music/delete/<int:song_id>', methods=['DELETE'])
def delete_song(song_id):
    try:
        songs = load_songs_db()
        song_to_delete = next((s for s in songs if s['id'] == song_id), None)
        
        if not song_to_delete:
            return jsonify({'status': 'error', 'message': 'Song not found'}), 404

        # Xác định thư mục gốc của ứng dụng
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # --- Xử lý xóa file audio ---
        audio_path = song_to_delete.get('filename', '')
        if audio_path:
            if audio_path.startswith('/'):
                audio_path = audio_path[1:]
            
            real_audio_path = os.path.normpath(os.path.join(base_dir, '..', audio_path))
            
            if os.path.exists(real_audio_path):
                try:
                    os.remove(real_audio_path)
                    print(f"Deleted audio: {real_audio_path}")
                except Exception as remove_error:
                    print(f"Failed to delete audio file {real_audio_path}: {remove_error}")
                    # Không ném lỗi ra ngoài → vẫn tiếp tục xóa DB
                else:
                    print(f"Audio not found: {real_audio_path}")

        # --- Xử lý xóa file cover ---
            cover_path = song_to_delete.get('cover', '')
            if cover_path and 'default.jpg' not in cover_path:
                if cover_path.startswith('/'):
                    cover_path = cover_path[1:]
                    
                real_cover_path = os.path.normpath(os.path.join(base_dir, '..', cover_path))
                
                if os.path.exists(real_cover_path):
                    try:
                        os.remove(real_cover_path)
                        print(f"Deleted cover: {real_cover_path}")
                    except Exception as remove_error:
                        print(f"Failed to delete cover file {real_cover_path}: {remove_error}")
                else:
                    print(f"Cover not found: {real_cover_path}")

        # --- Cập nhật database ---
        updated_songs = [s for s in songs if s['id'] != song_id]
        save_songs_db(updated_songs)
        
        print(f"Removed song ID {song_id} from database")
        return jsonify({'status': 'success', 'message': 'Song deleted successfully'})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Delete failed: {e}")
        return jsonify({'status': 'error', 'message': f'Delete failed: {str(e)}'}), 500

def get_file_path_from_url(url_path, default_folder='static'):
    """
    Chuyển đổi URL path thành file system path
    Ví dụ: /static/music/song.mp3 → static/music/song.mp3
    """
    if not url_path:
        return None
    
    # Bỏ dấu / đầu tiên nếu có
    if url_path.startswith('/'):
        return url_path[1:]
    
    # Nếu không có static/ ở đầu, thêm vào
    if not url_path.startswith('static/'):
        return os.path.join(default_folder, os.path.basename(url_path))
    
    return url_path

@music_bp.route('/api/music/<int:song_id>', methods=['GET'])
def get_song(song_id):
    """Lấy thông tin chi tiết một bài hát"""
    try:
        songs = load_songs_db()
        for song in songs:
            if song['id'] == song_id:
                return jsonify({
                    'status': 'success',
                    'song': song
                })
        
        return jsonify({
            'status': 'error',
            'message': 'Song not found'
        }), 404
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@music_bp.route('/api/music/update/<int:song_id>', methods=['PUT'])
def update_song(song_id):
    """Cập nhật thông tin bài hát"""
    try:
        songs = load_songs_db()
        song_to_update = None
        song_index = -1
        
        for i, song in enumerate(songs):
            if song['id'] == song_id:
                song_to_update = song
                song_index = i
                break
        
        if not song_to_update:
            return jsonify({
                'status': 'error',
                'message': 'Song not found'
            }), 404
        
        # Cập nhật thông tin
        data = request.get_json()
        if 'title' in data:
            song_to_update['title'] = data['title']
        if 'artist' in data:
            song_to_update['artist'] = data['artist']
        
        songs[song_index] = song_to_update
        save_songs_db(songs)
        
        return jsonify({
            'status': 'success',
            'message': 'Song updated successfully',
            'song': song_to_update
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Update failed: {str(e)}'
        }), 500

# Khởi tạo database với một số bài hát mẫu
def init_default_songs():
    """Khởi tạo database với bài hát mặc định nếu chưa có"""
    if not os.path.exists(SONGS_DB):
        default_songs = [
            {
                'id': 1,
                'title': 'Alert Sound 1',
                'artist': 'System',
                'file': '/static/music/alert1.mp3',
                'cover': r'D:\\python_code\\Individual_model\\BTLPYTHON\\FE\\statics\\covers\\default.jpg',
                'upload_date': datetime.now().isoformat(),
                'size': 0
            }
        ]
        save_songs_db(default_songs)
        print("✅ Initialized default songs database")