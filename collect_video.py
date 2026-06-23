import cv2
import mss
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import keyboard

try:
    import dxcam
    DXCAM_AVAILABLE = True
except:
    DXCAM_AVAILABLE = False

# =====================================================
# НАСТРОЙКИ
# =====================================================

OUTPUT_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\extracted_frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Настройки записи
FPS = 30
RECORD_HOTKEY = 'r'
EXTRACT_HOTKEY = 'e'
STOP_HOTKEY = 's'  # Отдельная клавиша для остановки

print("=" * 60)
print("VIDEO CAPTURE FOR NEURAL NETWORK")
print("=" * 60)
print(f"\nHotkeys:")
print(f"  {RECORD_HOTKEY.upper()} - Start recording")
print(f"  {STOP_HOTKEY.upper()} - Stop recording")
print(f"  {EXTRACT_HOTKEY.upper()} - Extract frames from last video")
print(f"  ESC - Exit")
print(f"\nVideos will be saved to: {OUTPUT_DIR}")
print(f"Extracted frames will be saved to: {FRAMES_DIR}")
print("=" * 60)

# =====================================================
# ИНИЦИАЛИЗАЦИЯ
# =====================================================

# Исправлено: mss.MSS вместо mss.mss()
sct = mss.mss()
monitor = sct.monitors[1]

use_dxcam = False
camera = None

if DXCAM_AVAILABLE:
    try:
        camera = dxcam.create(output_idx=0, output_color="BGR", fps=FPS)
        use_dxcam = camera is not None
        if use_dxcam:
            print("[+] Using dxcam for high FPS capture")
            camera.start(target_fps=FPS, video_mode=True)
    except:
        pass

if not use_dxcam:
    print("[*] Using mss")

def screenshot():
    if use_dxcam:
        frame = camera.get_latest_frame()
        if frame is not None:
            return frame
        time.sleep(0.001)
    
    frame = np.array(sct.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# =====================================================
# ВИДЕО ЗАПИСЬ
# =====================================================

class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.out = None
        self.frames = []
        self.start_time = 0
        
    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = OUTPUT_DIR / f"recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame = screenshot()
        h, w = frame.shape[:2]
        
        self.out = cv2.VideoWriter(str(self.video_path), fourcc, FPS, (w, h))
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        print(f"\n[RECORDING] Started: {self.video_path.name}")
        print(f"[RECORDING] Press {STOP_HOTKEY.upper()} to stop")
        
    def stop_recording(self):
        if self.out:
            self.out.release()
        duration = time.time() - self.start_time
        print(f"\n[RECORDING] Stopped!")
        print(f"[RECORDING] Duration: {duration:.1f} seconds")
        print(f"[RECORDING] Saved to: {self.video_path}")
        
        self.recording = False
        self.out = None
        return self.video_path
    
    def add_frame(self, frame):
        if self.recording and self.out:
            self.out.write(frame)
            self.frames.append(frame)
            
            # Показываем индикатор записи
            if len(self.frames) % 30 == 0:
                print(f"  Recording... {len(self.frames)} frames", end='\r')
    
    def extract_frames_from_video(self, video_path=None, step=5):
        """Извлекает кадры из видео с шагом step"""
        if video_path is None:
            videos = sorted(OUTPUT_DIR.glob("*.mp4"))
            if not videos:
                print("[!] No videos found!")
                return
            video_path = videos[-1]
        
        print(f"\n[EXTRACT] Extracting frames from: {video_path.name}")
        print(f"[EXTRACT] Step: every {step}th frame")
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_count = 0
        
        video_name = video_path.stem
        video_frames_dir = FRAMES_DIR / video_name
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                frame_filename = video_frames_dir / f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
                
                if saved_count % 50 == 0:
                    print(f"  Extracted {saved_count} frames...", end='\r')
            
            frame_count += 1
        
        cap.release()
        print(f"\n[EXTRACT] Complete!")
        print(f"[EXTRACT] Total frames: {frame_count}")
        print(f"[EXTRACT] Extracted frames: {saved_count}")
        print(f"[EXTRACT] Saved to: {video_frames_dir}")
        
        return video_frames_dir, saved_count


# =====================================================
# ОСНОВНОЙ ЦИКЛ
# =====================================================

recorder = VideoRecorder()
last_video_path = None
recording = False

print("\n[READY] Press R to start recording, S to stop, E to extract frames, ESC to exit\n")

while True:
    frame = screenshot()
    
    # Показываем предпросмотр
    preview = cv2.resize(frame, (320, 180))
    
    # Индикатор записи
    if recorder.recording:
        cv2.circle(preview, (20, 20), 10, (0, 0, 255), -1)
        cv2.putText(preview, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(preview, f"{len(recorder.frames)} frames", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Auction Bot - Video Capture", preview)
    
    # Добавляем кадр в запись
    if recorder.recording:
        recorder.add_frame(frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        if recorder.recording:
            last_video_path = recorder.stop_recording()
        break
    
    elif key == ord('r') or key == ord('R'):
        if not recorder.recording:
            recorder.start_recording()
    
    elif key == ord('s') or key == ord('S'):
        if recorder.recording:
            last_video_path = recorder.stop_recording()
    
    elif key == ord('e') or key == ord('E'):
        if last_video_path:
            recorder.extract_frames_from_video(last_video_path, step=3)
        else:
            recorder.extract_frames_from_video(step=3)

cv2.destroyAllWindows()
print("\n[OK] Program finished!")