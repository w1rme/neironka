import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
import random
import re
from collections import Counter
from sklearn.metrics import classification_report
import traceback
from torch.utils.data import Subset

from image_loading import safe_read_image
from model_config import MODEL_INPUT_SIZE

# =====================================================
# PATHS
# =====================================================

DATA_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka\data")
MODEL_DIR = Path(r"C:\Users\w1rmeee\Desktop\neironka")

MODEL_PATH = MODEL_DIR / "model_weights.pth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0006
RANDOM_SEED = 42

EARLY_STOPPING = 10
INIT_TRAINABLE_WEIGHTS = True
WEIGHT_INIT_METHOD = "kaiming"
USE_DYNAMIC_CLASS_WEIGHTS = True
GROUP_SPLIT_TIME_BUCKET_SECONDS = 10
LABEL_SMOOTHING = 0.05

# Full-frame mode: the model trains on the entire screenshot and
# only resizes it to the network input resolution.
USE_SMART_CROP = False

CLASSES = [
    "0_auction_house",
    "1_search_form",
    "2_no_auctions",
    "3_buy_menu",
    "4_confirm_buy",
    "5_buy_success",
    "6_buy_failed",
    "7_my_auction",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

NUM_CLASSES = len(CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[+] Device: {device}")

if torch.cuda.is_available():
    print(f"[+] GPU: {torch.cuda.get_device_name(0)}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)

# =====================================================
# SMART ROI CROP
# =====================================================

def smart_crop(img, label):

    if not USE_SMART_CROP:
        return img

    h, w = img.shape[:2]

    # popup screens
    if label in [3, 4, 5, 6]:

        margin_h = int(h * 0.30)
        margin_w = int(w * 0.30)

        return img[
            margin_h:h - margin_h,
            margin_w:w - margin_w
        ]

    # search/list screens
    if label in [0, 1, 2, 7]:

        top = int(h * 0.08)
        bottom = int(h * 0.92)

        left = int(w * 0.04)
        right = int(w * 0.96)

        return img[top:bottom, left:right]

    return img

# =====================================================
# DATASET
# =====================================================

class AuctionDataset(Dataset):

    def __init__(self, data_dir, classes, augment=True):

        self.classes = list(classes)
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.classes)
        }
        self.images = []
        self.labels = []

        self.augment = augment
        self.skipped_problem_files = 0

        print("[*] Loading dataset...")

        for class_name in self.classes:

            class_dir = data_dir / class_name

            if not class_dir.exists():
                print(f"[!] Missing: {class_dir}")
                continue

            loaded = 0

            for img_path in class_dir.glob("*.png"):

                img = safe_read_image(img_path)

                if img is None:
                    self.skipped_problem_files += 1
                    continue

                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

                loaded += 1

            print(f"[+] {class_name}: {loaded}")

        print(f"[+] Total images: {len(self.images)}")
        print(f"[+] Skipped problematic images: {self.skipped_problem_files}")

        counts = Counter(self.labels)
        self.class_counts = {
            idx: counts.get(idx, 0)
            for idx in range(NUM_CLASSES)
        }

        print("\n[DATASET BALANCE]")

        for class_name, idx in CLASS_TO_IDX.items():

            count = counts[idx]

            print(f"{class_name}: {count}")

            if count < 250:
                print("  WARNING: мало скринов")

    def __len__(self):
        return len(self.images)

    def get_class_weights(self):
        weights = []

        non_zero_counts = [
            count for count in self.class_counts.values()
            if count > 0
        ]

        if not non_zero_counts:
            return [1.0] * NUM_CLASSES

        # Use the average valid sample count as the reference so that:
        # - rarer classes get weights > 1
        # - more frequent classes get weights < 1
        # - classes near the dataset average stay close to 1
        reference_count = sum(non_zero_counts) / len(non_zero_counts)

        print(
            f"[+] Class weight formula: weight = {reference_count:.2f} / class_count"
        )

        for idx in range(NUM_CLASSES):
            count = self.class_counts.get(idx, 0)

            if count == 0:
                weights.append(0.0)
                continue

            weight = reference_count / count
            weights.append(weight)

            class_name = CLASSES[idx]
            print(
                f"[WEIGHT] {class_name}: "
                f"count={count} -> weight={weight:.4f}"
            )

        return weights

    def preprocess(self, img):
        img = cv2.resize(img, MODEL_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std

        return img

    def augment_img(self, img, label):

        if not self.augment:
            return img

        # brightness
        if random.random() > 0.5:
            brightness = random.uniform(0.75, 1.25)
            img = img * brightness
            img = np.clip(img, 0, 1)

        # noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.03, img.shape)
            img += noise
            img = np.clip(img, 0, 1)

        # blur
        if random.random() > 0.7:
            blur = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (blur, blur), 0)

        # jpeg artifacts
        if random.random() > 0.7:

            temp = (img * 255).astype(np.uint8)

            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                random.randint(40, 90)
            ]

            _, encimg = cv2.imencode('.jpg', temp, encode_param)

            temp = cv2.imdecode(encimg, 1)

            img = temp.astype(np.float32) / 255.0

        # stronger augment for success/fail
        if label in [5, 6]:

            if random.random() > 0.5:

                angle = random.uniform(-3, 3)

                rows, cols = img.shape[:2]

                matrix = cv2.getRotationMatrix2D(
                    (cols / 2, rows / 2),
                    angle,
                    1
                )

                img = cv2.warpAffine(
                    img,
                    matrix,
                    (cols, rows)
                )

        return img

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label = self.labels[idx]

        img = safe_read_image(img_path)

        if img is None:
            img = np.zeros(
                (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3),
                dtype=np.uint8
            )

        # SMART ROI
        img = smart_crop(img, label)

        img = self.preprocess(img)

        img = self.augment_img(img, label)

        tensor = torch.from_numpy(img).float().permute(2, 0, 1)

        return tensor, label


def build_grouped_split(dataset, train_ratio=0.8, seed=RANDOM_SEED):
    groups_by_class = {}

    for idx, img_path in enumerate(dataset.images):
        label = dataset.labels[idx]
        group_key = infer_capture_group(img_path)
        class_groups = groups_by_class.setdefault(label, {})
        class_groups.setdefault(group_key, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []
    total_group_count = 0

    print("[+] Grouped stratified split by class:")

    for label, class_name in enumerate(dataset.classes):
        class_groups = list(groups_by_class.get(label, {}).items())
        rng.shuffle(class_groups)
        total_group_count += len(class_groups)

        if not class_groups:
            print(f"    - {class_name}: no groups found")
            continue

        if len(class_groups) == 1:
            group_name, indices = class_groups[0]
            split_point = max(1, min(len(indices) - 1, int(len(indices) * train_ratio)))
            class_train_indices = indices[:split_point]
            class_val_indices = indices[split_point:]

            if not class_val_indices and class_train_indices:
                class_val_indices.append(class_train_indices.pop())
        else:
            val_group_count = max(1, int(round(len(class_groups) * (1 - train_ratio))))
            val_group_count = min(val_group_count, len(class_groups) - 1)

            val_group_items = class_groups[:val_group_count]
            train_group_items = class_groups[val_group_count:]

            class_train_indices = [
                idx
                for _, indices in train_group_items
                for idx in indices
            ]
            class_val_indices = [
                idx
                for _, indices in val_group_items
                for idx in indices
            ]

        train_indices.extend(class_train_indices)
        val_indices.extend(class_val_indices)

        print(
            f"    - {class_name}: groups={len(class_groups)} "
            f"train={len(class_train_indices)} val={len(class_val_indices)}"
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if not train_indices and val_indices:
        train_indices.append(val_indices.pop())

    if not val_indices and train_indices:
        val_indices.append(train_indices.pop())

    print(
        f"[+] Grouped split total: {total_group_count} groups | "
        f"train={len(train_indices)} val={len(val_indices)}"
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def infer_capture_group(img_path):
    parts = re.findall(r"\d+", Path(img_path).stem)

    for part in parts:
        if len(part) >= 9:
            bucket = int(part) // GROUP_SPLIT_TIME_BUCKET_SECONDS
            return f"{img_path.parent.name}_{bucket}"

    return f"{img_path.parent.name}_{Path(img_path).stem}"

# =====================================================
# MODEL
# =====================================================

class TransferLearningCNN(nn.Module):

    def __init__(self, num_classes=8):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

        if INIT_TRAINABLE_WEIGHTS:
            initialize_trainable_weights(
                self.backbone.fc,
                method=WEIGHT_INIT_METHOD
            )

    def forward(self, x):
        return self.backbone(x)


def initialize_trainable_weights(module, method="kaiming"):
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if method == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu"
                )

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

# =====================================================
# TRAIN
# =====================================================

def train_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(loader)} | Loss {loss.item():.4f}")

    accuracy = 100 * correct / total

    return total_loss / len(loader), accuracy

# =====================================================
# VALIDATE
# =====================================================

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    print("\n[CLASSIFICATION REPORT]")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0
        )
    )

    return total_loss / len(loader), accuracy

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FORZA AI TRAINER")
    print("=" * 60)
    print(f"[+] Model input size: {MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]}")
    print(f"[+] Batch size: {BATCH_SIZE}")

    if not DATA_DIR.exists():
        print("[ERROR] DATA DIR NOT FOUND")
        input()
        exit()

    try:

        dataset = AuctionDataset(
            DATA_DIR,
            CLASSES,
            augment=True
        )

        train_dataset, val_dataset = build_grouped_split(
            dataset,
            train_ratio=0.8,
            seed=RANDOM_SEED,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        model = TransferLearningCNN(NUM_CLASSES).to(device)

        if USE_DYNAMIC_CLASS_WEIGHTS:
            class_weights = torch.tensor(
                dataset.get_class_weights(),
                dtype=torch.float32,
                device=device
            )
            print(
                "[+] Dynamic class weights from folder image counts: "
                f"{class_weights.tolist()}"
            )
        else:
            class_weights = torch.ones(
                NUM_CLASSES,
                dtype=torch.float32,
                device=device
            )
            print("[+] Static class weights disabled, using all ones")

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        best_acc = 0
        patience_counter = 0

        for epoch in range(EPOCHS):

            print(f"\n===== EPOCH {epoch + 1}/{EPOCHS} =====")

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )

            val_loss, val_acc = validate(
                model,
                val_loader,
                criterion
            )

            scheduler.step(val_loss)

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Acc : {train_acc:.2f}%")

            print(f"Val Loss  : {val_loss:.4f}")
            print(f"Val Acc   : {val_acc:.2f}%")

            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    model.state_dict(),
                    MODEL_PATH
                )

                print(f"[+] BEST MODEL SAVED ({best_acc:.2f}%)")

                patience_counter = 0

            else:
                patience_counter += 1

                print(f"[*] No improvement ({patience_counter}/{EARLY_STOPPING})")

            if patience_counter >= EARLY_STOPPING:
                print("[STOP] Early stopping")
                break

        print("\n[TRAINING COMPLETE]")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print(f"Saved: {MODEL_PATH}")

    except Exception as e:

        print(f"[ERROR] {e}")

        traceback.print_exc()

    input("\nPress Enter to exit...")
