

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import os
import torch.nn as nn




class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet50(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = ResNet50()
        self.body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.out_channels = 2048

    def forward(self, x):
        return self.body(x)

def get_model(num_classes=21):  # 20 VOC classes + background
    backbone = ResNetBackbone()
    backbone.out_channels = 2048

    rpn_anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


# In[5]:


from torchvision.datasets import VOCDetection
dataset = VOCDetection(root="./", year="2007", image_set="train", download=True)


# In[6]:


import os

print("Folders inside ./VOCdevkit:")
print(os.listdir("./VOCdevkit"))

print("Folders inside ./VOCdevkit/VOC2007:")
print(os.listdir("./VOCdevkit/VOC2007"))


# In[7]:


import os
import urllib.request
import tarfile
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import xml.etree.ElementTree as ET

# 1. Download + Extract VOC2007 if not done already

data_dir = "./VOCdevkit"
voc_tar_path = "./VOCtrainval_06-Nov-2007.tar"
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"

if not os.path.exists(voc_tar_path):
    print("Downloading VOC2007 dataset...")
    urllib.request.urlretrieve(VOC_URL, voc_tar_path)
    print("Download complete.")

# Extract dataset if not extracted
if not os.path.exists(os.path.join(data_dir, "VOCdevkit", "VOC2007")):
    print("Extracting VOC2007 dataset...")
    with tarfile.open(voc_tar_path) as tar:
        tar.extractall(path=data_dir)  # this creates VOCdevkit/VOCdevkit/VOC2007
    print("Extraction complete.")

# 2. Define dataset class

VOC_CLASSES = [
    "__background__",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
class_name_to_idx = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        ann_path = os.path.join(self.annotation_dir, img_name.replace(".jpg", ".xml"))
        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name not in class_name_to_idx:
                continue
            label = class_name_to_idx[cls_name]
            labels.append(label)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

# 3. Update paths to match your folder structure:

image_dir = "./VOCdevkit/VOCdevkit/VOC2007/JPEGImages"
annotation_dir = "./VOCdevkit/VOCdevkit/VOC2007/Annotations"

dataset = PascalVOCDataset(image_dir, annotation_dir)

def collate_fn(batch):
    return tuple(zip(*batch))

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 4. Test by loading one batch

images, targets = next(iter(data_loader))
print(f"Loaded batch with {len(images)} images")
print(f"Image[0] shape: {images[0].shape}")
print(f"Target[0] keys: {targets[0].keys()}")


# In[8]:


from torchvision.ops import box_iou


# In[9]:


def evaluate_model(model, data_loader, device, iou_threshold=0.5, max_batches=5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= max_batches:
                break
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].cpu()
                gt_boxes = target["boxes"].cpu()

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched = (ious >= iou_threshold).any(dim=1)
                correct += matched.sum().item()
                total += len(gt_boxes)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy



# In[10]:


from tqdm import tqdm
import torch
import torchvision

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 21  # VOC classes (20 + background)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    loop = tqdm(data_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")

    for images, targets in loop:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        avg_loss = running_loss / (loop.n + 1)
        loop.set_postfix(loss=avg_loss)

    lr_scheduler.step

    # Evaluate training accuracy (IoU ≥ 0.5)
    acc = evaluate_model(model, data_loader, device)
    print(f"📊 Epoch {epoch+1}: Training Accuracy (IoU ≥ 0.5) = {acc:.2f}%\n")


print("Training complete.")


# In[13]:


get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py')


# In[15]:


get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py')


# In[17]:


get_ipython().system('pip install -q fvcore')


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms

# Load label map
idx_to_class = {v: k for k, v in class_name_to_idx.items()}

# Function to visualize predictions
def visualize_prediction(image, boxes, labels, scores, threshold=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, f"{idx_to_class[label]}: {score:.2f}", 
                color='white', fontsize=12, backgroundcolor='red')

    plt.axis('off')
    plt.show()

# Load model
model.eval()

# Test on a few samples from the test loader
num_test_images = 3
for i, (images, targets) in enumerate(data_loader):
    if i >= num_test_images:
        break

    images = [img.to(device) for img in images]

    with torch.no_grad():
        outputs = model(images)

    for img, output in zip(images, outputs):
        boxes = output['boxes'].cpu()
        labels = output['labels'].cpu()
        scores = output['scores'].cpu()

        # Optional: apply Non-Maximum Suppression (NMS)
        keep = nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        print("Number of boxes:", len(boxes))
        print("Scores:", scores)
        print("Detected boxes:", boxes.shape[0])
        print("Scores:", scores[:10])  # Print top 10 scores



        visualize_prediction(img.cpu(), boxes, labels, scores, threshold=0.2)


# In[35]:


# Ensure model is in evaluation mode
model.eval()

# Evaluate and print test accuracy
test_accuracy = evaluate_model(model, data_loader, device, iou_threshold=0.5, max_batches=10)  # You can increase max_batches for more coverage
print(f"✅ Test Accuracy (IoU ≥ 0.5): {test_accuracy:.2f}%")


from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Initialize metric
metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])

# Collect predictions and ground truths
model.eval()
with torch.no_grad():
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        # Convert to required format
        preds = []
        gts = []
        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].cpu(),
                "scores": output["scores"].cpu(),
                "labels": output["labels"].cpu()
            })
            gts.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu()
            })

        metric.update(preds, gts)

# Compute mAP
results = metric.compute()
print(f"📊 mAP@0.5: {results['map_50']:.4f}")
print(f"📊 mAP@[.5:.95]: {results['map']:.4f}")


# In[47]:


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch

# Dummy input
model.eval()
dummy_input = [torch.randn(3, 300, 300).to(device)]

# FLOP analysis
flops = FlopCountAnalysis(model, dummy_input)
params = parameter_count(model)

print(f"🧮 Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
print(f"📦 Total Parameters: {params[''] / 1e6:.2f} Million")


# In[ ]:




