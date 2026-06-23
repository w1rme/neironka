import time
import cv2
import mss
import numpy as np
from pathlib import Path
import keyboard

try:
    import dxcam
    DXCAM_AVAILABLE = True
except:
    DXCAM_AVAILABLE = False


DATA_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\data")
CLASSES = {
    "0_auction_house": "Главное меню аукциона",
    "1_search_form": "Форма поиска",
    "2_no_auctions": "Аукционов нет",
    "3_buy_menu": "Меню покупки",
    "4_confirm_buy": "Подтверждение выкупа",
    "5_buy_success": "Успешный выкуп",
    "6_buy_failed": "Ошибка выкупа",
    "7_my_auction": "Мой размещённый лот",
}

for class_dir in CLASSES.keys():
    (DATA_DIR / class_dir).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SCREENSHOT COLLECTOR FOR NEURAL NETWORK")
print("=" * 60)
print("\nHotkeys:")
for key, desc in CLASSES.items():
    print(f"  {key[0]} - {desc}")
print("  BACKSPACE - Undo last save")
print("  ESC - Exit and save")
print("=" * 60)


sct = mss.mss()
monitor = sct.monitors[1]

use_dxcam = False
camera = None

if DXCAM_AVAILABLE:
    try:
        camera = dxcam.create(output_idx=0, output_color="BGR")
        use_dxcam = camera is not None
        if use_dxcam:
            print("[+] Using dxcam (fast screenshots)")
    except:
        pass

if not use_dxcam:
    print("[*] Using mss")

def screenshot():
    if use_dxcam:
        frame = camera.grab()
        if frame is not None:
            return frame
        time.sleep(0.001)
        frame = camera.grab()
        if frame is not None:
            return frame
    
    frame = np.array(sct.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

counters = {key: 0 for key in CLASSES.keys()}
last_saved = []  

print("\n[READY] Press hotkeys to capture screenshots...")
print("[TIP] Take 50-100 screenshots of each screen type")
print("[TIP] Press BACKSPACE to undo last save\n")

while True:
    frame = screenshot()
    
    preview = cv2.resize(frame, (320, 180))
    cv2.imshow("Capture Preview - Press hotkey to save", preview)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        for class_dir, count in counters.items():
            print(f"  {class_dir}: {count} screenshots")
        print("=" * 60)
        break
    

    if key == 8:  
        if last_saved:
            class_dir, file_path = last_saved.pop()
            try:
                file_path.unlink()  
                counters[class_dir] -= 1
                print(f"[UNDO] Deleted last save: {class_dir} (now {counters[class_dir]})")
                print("\a", end="")
            except Exception as e:
                print(f"[!] Failed to delete {file_path.name}: {e}")
        else:
            print("[UNDO] Nothing to undo")
        continue
    
    for i, class_dir in enumerate(CLASSES.keys()):
        if key == ord(str(i)):
            filename = DATA_DIR / class_dir / f"{int(time.time())}_{counters[class_dir]}.png"
            cv2.imwrite(str(filename), frame)
            counters[class_dir] += 1
            last_saved.append((class_dir, filename))
            print(f"[+] Saved to {class_dir} (total: {counters[class_dir]})")
            print("\a", end="")
            break

cv2.destroyAllWindows()
print("\n[OK] Data collection complete!")