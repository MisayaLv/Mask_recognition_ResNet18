#train2 解冻backbone微调

from collections import Counter

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

#transform配置
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#准备数据集
train_dataset = ImageFolder('../data/train', transform=train_transform)
val_dataset = ImageFolder('../data/val', transform=val_transform)

#利用dataloader加载数据集
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

#加载预训练ResNet
model = models.resnet18(pretrained=True)

#解冻最后一层
for param in model.layer4.parameters():
    param.requires_grad = True

for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

#修改分类头 满足项目的三分类
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)

#确认只有fc层在训练
#params_to_update = [p for p in model.parameters() if p.requires_grad]
#print(len(params_to_update))

#损失函数
criterion = nn.CrossEntropyLoss()

#优化器
params_to_update = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params_to_update, lr=1e-4)

#设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#单轮训练函数
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

#验证函数（无梯度）
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    #预测及标签
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        #保存预测及标签
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

#正式训练
num_epochs = 15

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )

    #训练集、验证集正确率
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    # #统计错误样本数量
    # val_preds = np.array(val_preds)
    # val_labels = np.array(val_labels)
    #
    # error_mask = val_preds != val_labels
    # num_errors = error_mask.sum()
    #
    # print(f"Val Errors: {num_errors} / {len(val_labels)}")
    #
    # #统计错误类别
    # error_labels = val_labels[error_mask]
    # counter = Counter(error_labels)
    #
    # print("错误样本类别分布：", counter)

    #仅最后一轮打印
    if epoch == num_epochs - 1:
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        error_mask = val_preds != val_labels
        print(f"Final Val Errors: {error_mask.sum()} / {len(val_labels)}")

        print("Error label distribution:",
              Counter(val_labels[error_mask]))
