import cv2
import shutil
from pathlib import Path
import time

FRAMES_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\extracted_frames")
DATA_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\data")

CLASSES = {
    '0': '0_auction_house',
    '1': '1_search_form',
    '2': '2_no_auctions',
    '3': '3_buy_menu',
    '4': '4_confirm_buy',
    '5': '5_buy_success',
    '6': '6_buy_failed',
    '7': '7_my_auction',
}

for class_folder in CLASSES.values():
    (DATA_DIR / class_folder).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FRAME CLASSIFIER")
print("=" * 60)
print(f"\nFrames directory: {FRAMES_DIR}")
print("\nHotkeys for classification:")
for key, class_name in CLASSES.items():
    print(f"  {key} - {class_name}")
print("  SPACE - Next frame")
print("  BACKSPACE - Previous frame")
print("  ESC - Save and exit")
print("=" * 60)

all_frames = []
for video_dir in sorted(FRAMES_DIR.iterdir()):
    if video_dir.is_dir():
        for frame_path in sorted(video_dir.glob("*.png")):
            all_frames.append(frame_path)

if not all_frames:
    print("\n[ERROR] No frames found!")
    print("Please record a video first using collect_video.py")
    exit()

print(f"\n[INFO] Found {len(all_frames)} frames to classify")

current_idx = 0
classified = {class_name: 0 for class_name in CLASSES.values()}
skipped = []

cv2.namedWindow("Classify Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classify Frame", 800, 600)

print("\n[READY] Start classifying frames...\n")

while current_idx < len(all_frames):
    frame_path = all_frames[current_idx]
    img = cv2.imread(str(frame_path))
    
    if img is None:
        print(f"[!] Cannot read: {frame_path.name}")
        current_idx += 1
        continue
    
    info = f"Frame {current_idx + 1}/{len(all_frames)}: {frame_path.parent.name}/{frame_path.name}"
    cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Classify Frame", img)
    
    key = cv2.waitKey(0) & 0xFF
    
    if key == 27:
        print("\n[Saving] Progress saved...")
        break
    
    elif key == ord(' '):
        current_idx += 1
        continue
    
    elif key == 8:
        if current_idx > 0:
            current_idx -= 1
        continue
    
    elif chr(key) in CLASSES:
        class_key = chr(key)
        class_folder = CLASSES[class_key]
        dest_dir = DATA_DIR / class_folder
        
        new_name = f"{class_folder}_{int(time.time())}_{current_idx}.png"
        dest_path = dest_dir / new_name
        shutil.copy2(str(frame_path), str(dest_path))
        
        classified[class_folder] += 1
        print(f"[+] Classified as {class_folder} (Total: {classified[class_folder]})")
        
        current_idx += 1
    
    elif key == ord('d'):
        skipped.append(frame_path.name)
        print(f"[?] Skipped (difficult): {frame_path.name}")
        current_idx += 1

cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("CLASSIFICATION STATISTICS")
print("=" * 60)

for class_name, count in classified.items():
    bar = "█" * (count // 2) if count > 0 else "░"
    print(f"{class_name:20s}: {count:3d} {bar}")

print(f"\nSkipped frames: {len(skipped)}")
print(f"Total classified: {sum(classified.values())}")
print("=" * 60)

print("\n[OK] Classification complete!")
print("Now run train_model.py to train the neural network!")