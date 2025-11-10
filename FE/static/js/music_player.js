// Music Player JavaScript
let songs = [];
let currentSongIndex = -1;
let isPlaying = false;

// DOM Elements
const audioPlayer = document.getElementById('audioPlayer');
const playBtn = document.getElementById('playBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');
const currentTimeEl = document.getElementById('currentTime');
const durationEl = document.getElementById('duration');
const volumeSlider = document.getElementById('volumeSlider');
const volumeValue = document.getElementById('volumeValue');
const muteBtn = document.getElementById('muteBtn');
const songTitle = document.getElementById('songTitle');
const songArtist = document.getElementById('songArtist');
const albumArt = document.getElementById('albumArt');
const playlist = document.getElementById('playlist');
const uploadBtn = document.getElementById('uploadBtn');
const uploadModal = document.getElementById('uploadModal');
const closeModal = document.getElementById('closeModal');
const cancelBtn = document.getElementById('cancelBtn');
const submitBtn = document.getElementById('submitBtn');
const audioFileInput = document.getElementById('audioFileInput');
const coverFileInput = document.getElementById('coverFileInput');
const audioFileName = document.getElementById('audioFileName');
const coverFileName = document.getElementById('coverFileName');
const songTitleInput = document.getElementById('songTitleInput');
const artistInput = document.getElementById('artistInput');

// Initialize
async function init() {
    await loadSongs();
    setupEventListeners();
    audioPlayer.volume = volumeSlider.value / 100;
}

// Load songs from API
async function loadSongs() {
    try {
        const response = await fetch('/api/music/list');
        const data = await response.json();
        console.log('Load songs response:', data);
        if (data.status === 'success') {
            songs = data.songs;
            renderPlaylist();
            console.log('Songs loaded:', songs);
        }
    } catch (error) {
        console.error('Error loading songs:', error);
    }
}

// Render playlist
function renderPlaylist() {
    playlist.innerHTML = '';
    songs.forEach((song, index) => {
        const songItem = document.createElement('div');
        songItem.className = 'song-item';
        if (index === currentSongIndex) {
            songItem.classList.add('active');
        }
        
        songItem.innerHTML = `
            <div class="song-thumbnail">
                <img src="${song.cover}" alt="${song.title}" onerror="this.src='/static/covers/default.jpg'">
            </div>
            <div class="song-details">
                <div class="song-name">${song.title}</div>
                <div class="song-creator">${song.artist}</div>
            </div>
            <button class="delete-btn" onclick="deleteSong(${song.id})" title="Delete">✕</button>
        `;
        
        songItem.addEventListener('click', (e) => {
            if (!e.target.classList.contains('delete-btn')) {
                playSong(index);
            }
        });
        
        playlist.appendChild(songItem);
    });
}

// Play song
function playSong(index) {
    if (index < 0 || index >= songs.length) return;
    
    currentSongIndex = index;
    const song = songs[index];
    
    audioPlayer.src = song.filename;
    songTitle.textContent = song.title;
    songArtist.textContent = song.artist;
    albumArt.src = song.cover;
    
    audioPlayer.play();
    isPlaying = true;
    updatePlayButton();
    renderPlaylist();
}

// Toggle play/pause
function togglePlayPause() {
    if (songs.length === 0) {
        alert('No songs in playlist!');
        return;
    }
    
    if (currentSongIndex === -1) {
        playSong(0);
        return;
    }
    
    if (isPlaying) {
        audioPlayer.pause();
    } else {
        audioPlayer.play();
    }
    isPlaying = !isPlaying;
    updatePlayButton();
}

// Update play button icon
function updatePlayButton() {
    if (isPlaying) {
        playBtn.innerHTML = `
            <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
            </svg>
        `;
    } else {
        playBtn.innerHTML = `
            <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
            </svg>
        `;
    }
}

// Previous song
function playPrevious() {
    if (currentSongIndex > 0) {
        playSong(currentSongIndex - 1);
    } else {
        playSong(songs.length - 1);
    }
}

// Next song
function playNext() {
    if (currentSongIndex < songs.length - 1) {
        playSong(currentSongIndex + 1);
    } else {
        playSong(0);
    }
}

// Format time
function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Update progress
function updateProgress() {
    const percent = (audioPlayer.currentTime / audioPlayer.duration) * 100;
    progressFill.style.width = `${percent}%`;
    currentTimeEl.textContent = formatTime(audioPlayer.currentTime);
    durationEl.textContent = formatTime(audioPlayer.duration);
}

// Seek
function seek(e) {
    const rect = progressBar.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    audioPlayer.currentTime = percent * audioPlayer.duration;
}

// Volume control
function updateVolume() {
    const volume = volumeSlider.value / 100;
    audioPlayer.volume = volume;
    volumeValue.textContent = `${volumeSlider.value}%`;
    
    if (volume === 0) {
        muteBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
            </svg>
        `;
    } else {
        muteBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
            </svg>
        `;
    }
}

// Toggle mute
function toggleMute() {
    if (audioPlayer.volume > 0) {
        audioPlayer.dataset.previousVolume = audioPlayer.volume;
        audioPlayer.volume = 0;
        volumeSlider.value = 0;
    } else {
        const prevVolume = parseFloat(audioPlayer.dataset.previousVolume) || 0.7;
        audioPlayer.volume = prevVolume;
        volumeSlider.value = prevVolume * 100;
    }
    updateVolume();
}

// Delete song
async function deleteSong(songId) {
    if (!confirm('Are you sure you want to delete this song?')) return;
    
    try {
        const response = await fetch(`/api/music/delete/${songId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            await loadSongs();
            alert('Song deleted successfully!');
        } else {
            alert('Failed to delete song: ' + data.message);
        }
    } catch (error) {
        console.error('Error deleting song:', error);
        alert('Error deleting song!');
    }
}

// Show upload modal
function showUploadModal() {
    uploadModal.classList.add('active');
}

// Hide upload modal
function hideUploadModal() {
    uploadModal.classList.remove('active');
    songTitleInput.value = '';
    artistInput.value = '';
    audioFileInput.value = '';
    coverFileInput.value = '';
    audioFileName.textContent = '';
    coverFileName.textContent = '';
}

// Handle file selection
function handleFileSelect(input, displayElement) {
    if (input.files && input.files[0]) {
        displayElement.textContent = `✓ ${input.files[0].name}`;
    } else {
        displayElement.textContent = '';
    }
}

// Upload song
async function uploadSong() {
    const title = songTitleInput.value.trim();
    const artist = artistInput.value.trim();
    const audioFile = audioFileInput.files[0];
    
    if (!title || !artist || !audioFile) {
        alert('Please fill in all required fields!');
        return;
    }
    
    const formData = new FormData();
    formData.append('title', title);
    formData.append('artist', artist);
    formData.append('audio', audioFile);
    
    if (coverFileInput.files[0]) {
        formData.append('cover', coverFileInput.files[0]);
    }
    
    try {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Uploading...';
        
        const response = await fetch('/api/music/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert('✅ Song uploaded successfully!');
            hideUploadModal();
            await loadSongs();
        } else {
            alert('❌ Upload failed: ' + data.message);
        }
    } catch (error) {
        console.error('Error uploading song:', error);
        alert('❌ Upload error!');
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Upload';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Player controls
    playBtn.addEventListener('click', togglePlayPause);
    prevBtn.addEventListener('click', playPrevious);
    nextBtn.addEventListener('click', playNext);
    
    // Audio events
    audioPlayer.addEventListener('timeupdate', updateProgress);
    audioPlayer.addEventListener('ended', playNext);
    audioPlayer.addEventListener('loadedmetadata', () => {
        durationEl.textContent = formatTime(audioPlayer.duration);
    });
    
    // Progress bar
    progressBar.addEventListener('click', seek);
    
    // Volume
    volumeSlider.addEventListener('input', updateVolume);
    muteBtn.addEventListener('click', toggleMute);
    
    // Upload modal
    uploadBtn.addEventListener('click', showUploadModal);
    closeModal.addEventListener('click', hideUploadModal);
    cancelBtn.addEventListener('click', hideUploadModal);
    submitBtn.addEventListener('click', uploadSong);
    
    // File inputs
    audioFileInput.addEventListener('change', () => {
        handleFileSelect(audioFileInput, audioFileName);
    });
    coverFileInput.addEventListener('change', () => {
        handleFileSelect(coverFileInput, coverFileName);
    });
    
    // Close modal on outside click
    uploadModal.addEventListener('click', (e) => {
        if (e.target === uploadModal) {
            hideUploadModal();
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);