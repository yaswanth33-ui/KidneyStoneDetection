import torch
import torch.nn as nn
from tqdm import tqdm

# ------------------ Base Building Blocks ------------------
class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.use_shortcut = c1 == c2
        if self.use_shortcut:
            self.conv = ConvBNSiLU(c1, c2, k=1)
        else:
            self.conv1 = ConvBNSiLU(c1, c2, k=1)
            self.conv2 = ConvBNSiLU(c2, c2, k=3)

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x)
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return torch.cat([x, x2], dim=1)

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class C3Block(nn.Module):
    def __init__(self, in_ch, out_ch, n=3):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_ch, in_ch, k=1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(in_ch, in_ch) for _ in range(n)])
        self.concat_conv = ConvBNSiLU(in_ch * 2, out_ch, k=1)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.bottlenecks(y1)
        cat = torch.cat((y1, y2), dim=1)
        out = self.concat_conv(cat)
        return self.se(out)

# ------------------ YOLOv5m Architecture (Kidney Stone Detection) ------------------
class YOLOv5mSE(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Reduced channel dimensions
        self.stem = ConvBNSiLU(3, 24, k=3, s=1)  # Reduced from 32
        self.stage1 = C3Block(24, 48, n=1)       # Reduced from 64
        self.stage2 = C3Block(48, 96, n=2)       # Reduced from 128, n=2 instead of 3
        self.stage3 = C3Block(96, 192, n=2)      # Reduced from 256, n=2 instead of 3
        self.stage4 = C3Block(192, 384, n=1)     # Reduced from 512

        # Neck with reduced channels
        self.neck1 = ConvBNSiLU(384, 192, k=1)   # Reduced from 256
        self.neck2 = C3Block(192 + 192, 192, n=1)  # Reduced from 256

        self.neck3 = ConvBNSiLU(192, 96, k=1)    # Reduced from 128
        self.neck4 = C3Block(96 + 96, 96, n=1)   # Reduced from 128

        # Head with reduced channels
        self.detect1 = nn.Conv2d(96, (num_classes + 5) * 3, 1)   # Reduced from 128
        self.detect2 = nn.Conv2d(192, (num_classes + 5) * 3, 1)  # Reduced from 256
        self.detect3 = nn.Conv2d(384, (num_classes + 5) * 3, 1)  # Reduced from 512

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # Neck with explicit size matching
        n1 = self.neck1(x4)
        n1_up = nn.functional.interpolate(n1, size=x3.shape[2:], mode='nearest')
        n2 = self.neck2(torch.cat([n1_up, x3], dim=1))

        n2_up = nn.functional.interpolate(self.neck3(n2), size=x2.shape[2:], mode='nearest')
        n3 = self.neck4(torch.cat([n2_up, x2], dim=1))

        # Head
        out1 = self.detect1(n3)
        out2 = self.detect2(n2)
        out3 = self.detect3(x4)

        return out1, out2, out3
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import os
import cv2
import numpy as np
from PIL import Image

# -------------------- Dataset with Bilateral Filtering + Augmentation --------------------
class KidneyStoneDataset(Dataset):
    def __init__(self, split_dir, img_size=320):
        self.image_dir = os.path.join(split_dir, "images")
        self.label_dir = os.path.join(split_dir, "labels")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.img_size = img_size
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor()
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        # Read image and apply bilateral filter
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = self.transforms(img)

        # Load labels (YOLO format)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    targets.append([cls, x, y, w, h])

        return img_tensor, torch.tensor(targets), img_name


# -------------------- Custom YOLO Loss --------------------
def bbox_ciou(box1, box2):
    # Compute Complete IoU (CIoU)
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area - inter_area
    iou = inter_area / (union + 1e-6)

    # center distance
    center_dist = (box1[..., 0] - box2[..., 0]) ** 2 + (box1[..., 1] - box2[..., 1]) ** 2
    # diagonal length of the smallest enclosing box
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(box1[..., 2] / box1[..., 3]) - torch.atan(box2[..., 2] / box2[..., 3]), 2)
    alpha = v / (1 - iou + v + 1e-6)
    ciou = iou - center_dist / (enc_diag + 1e-6) - alpha * v
    return ciou


class YoloLoss(torch.nn.Module):
    def __init__(self, lambda_cls=1.0, lambda_obj=1.0, lambda_box=5.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # Decode preds
        if isinstance(preds, tuple):
            preds = torch.cat([p.view(p.size(0), -1, p.size(-1)) for p in preds], dim=1)

        batch_size = preds.size(0)
        loss_cls, loss_obj, loss_box = 0.0, 0.0, 0.0

        for b in range(batch_size):
            pred = preds[b]  # [N, 6] or [N, 85] -> assuming [x, y, w, h, obj, cls...]
            t = targets[b]   # [M, 5] -> [cls, x, y, w, h]

            if len(t) == 0:
                continue

            for gt in t:
                gt_cls, gx, gy, gw, gh = gt.to(pred.device)
                gt_box = torch.tensor([gx, gy, gw, gh], device=pred.device)

                ious = bbox_ciou(pred[:, :4], gt_box.unsqueeze(0)).squeeze()
                best_idx = torch.argmax(ious)

                pred_box = pred[best_idx, :4]
                pred_obj = pred[best_idx, 4]
                pred_cls = pred[best_idx, 5:]

                ciou = bbox_ciou(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
                loss_box += 1 - ciou
                loss_obj += self.bce(pred_obj, torch.tensor(1.0, device=pred.device))
                loss_cls += self.bce(pred_cls[int(gt_cls)], torch.tensor(1.0, device=pred.device))

        total_loss = self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls
        return total_loss



# -------------------- Training Function --------------------
def train(model, train_loader, optimizer, epochs):
    model.train()
    criterion = YoloLoss()  # Create the loss function instance
    
    for epoch in range(epochs):
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(len(train_loader)), 
                         desc=f'Epoch {epoch+1}/{epochs}',
                         position=0)
        
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (imgs, targets, img_names) in enumerate(train_loader):
            imgs = imgs.cuda()
            targets = [t.cuda() for t in targets]
            
            outputs = model(imgs)
            loss = criterion(outputs, targets)  # Use the YoloLoss instance
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()
            
            total_loss += loss.item()  # Use .item() to get the float value for tracking
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar with current loss
            epoch_pbar.set_postfix({
                'loss': f'{total_loss/(i+1):.4f}',
                'batch': f'{i+1}/{len(train_loader)}'
            })
            epoch_pbar.update(1)
        
        epoch_pbar.close()
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}')

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, targets, _ in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets
    """
    images = []
    targets = []
    img_names = []
    
    for img, target, img_name in batch:
        images.append(img)
        targets.append(target)
        img_names.append(img_name)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images)
    
    return images, targets, img_names

# -------------------- Main Script --------------------
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath('__file__')), '..', 'dataset', 'train')
    dataset = KidneyStoneDataset(path)
    
    # Reduce batch size to save memory
    batch_size = 1  # Reduced from 2
    accumulation_steps = 8  # Increased from 4 to maintain effective batch size
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep this at 0 to prevent memory issues
        collate_fn=custom_collate_fn
    )

    # Enable memory efficient mode
    torch.cuda.empty_cache()  # Clear any unused memory
    
    model = YOLOv5mSE(num_classes=1).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    criterion = YoloLoss()

    # Training with progress bars
    train(model, train_loader, optimizer, epochs=2)

