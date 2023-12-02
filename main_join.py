import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from datasets.s_coco_set import CustomDataset
import numpy as np

import rpn.nms as nms
import rpn.ss as ss
import roi_align.roi_align as roi_align
import mask.mask as mask
import cls_bbox.cls_bbox as Classifier

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create custom dataset and dataloader
root_folder = 'datasets/'
custom_dataset = CustomDataset(root_folder, transform=transform)
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)

datas=[]
for inputs, annotations in dataloader:
    datas.append((inputs,annotations))
    if len(datas) >= 1:
        break


'''inputs data & annotations [(tensor([[[[-1.7754, -1.8268, -1.9124,  ..., -0.7822, -0.8507, -0.9192],
          [-1.7925, -1.8439, -1.8953,  ..., -0.7822, -0.8164, -0.8678],
          [-1.8610, -1.8610, -1.8782,  ..., -0.7822, -0.7650, -0.7479],
          ...,
          [-1.8782, -1.9295, -1.9980,  ..., -1.6213, -1.5870, -1.5699],
          [-1.8439, -1.8953, -1.9638,  ..., -1.6042, -1.5357, -1.4672],
          [-1.8268, -1.8782, -1.9467,  ..., -1.6042, -1.5014, -1.4158]],

         [[-1.8431, -1.8957, -1.9482,  ..., -0.3025, -0.3025, -0.3025],
          [-1.7556, -1.8081, -1.8782,  ..., -0.3025, -0.3025, -0.3200],
          [-1.5455, -1.6331, -1.7381,  ..., -0.2850, -0.3200, -0.3375],
          ...,
          [-1.5630, -1.5455, -1.5280,  ..., -1.5105, -1.5105, -1.5105],
          [-1.5805, -1.5455, -1.5105,  ..., -1.4930, -1.5105, -1.5280],
          [-1.5805, -1.5455, -1.5105,  ..., -1.4930, -1.5105, -1.5280]],

         [[-1.5256, -1.6302, -1.7522,  ...,  0.3045,  0.3742,  0.4439],
          [-1.3687, -1.4733, -1.6127,  ...,  0.3045,  0.3742,  0.4265],
          [-1.0027, -1.1247, -1.3164,  ...,  0.3219,  0.3742,  0.4091],
          ...,
          [-0.8284, -0.8284, -0.8110,  ..., -1.2816, -1.3339, -1.3861],
          [-0.8458, -0.8284, -0.8110,  ..., -1.3687, -1.3513, -1.3339],
          [-0.8458, -0.8284, -0.8110,  ..., -1.4036, -1.3513, -1.2990]]]])
'''



#===============[feature map]===============================

# ResNet-50 모델 불러오기
resnet50_model = resnet50(pretrained=True)

# 4번째 레이어까지의 모델 정의
model_up_to_layer4 = torch.nn.Sequential(*list(resnet50_model.children())[:-2])

# 모델에 이미지 전달하여 특성 맵 얻기
with torch.no_grad():
    model_up_to_layer4.eval()
    features = model_up_to_layer4(inputs)

#===============[RoI]======================================

def ch_box_shape(boxes):
    re_box = []
    for i in boxes:
        x, y, w, h = i
        re_box.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
    return re_box

img_ss = inputs.squeeze(0).numpy().transpose(1, 2, 0)
rects = ss.selective_search(img_ss)
boxes, idx = nms.nms(rects)
roi_boxes = ch_box_shape(boxes)

rois=[]
for tpl, val in zip(roi_boxes, idx):
    rois.append([val] + list(tpl))
rois_tensor = torch.as_tensor(rois, dtype=torch.float)
    
valid_rois = rois_tensor[:, 0] < features.size(0)
filtered_rois_tensor = rois_tensor[valid_rois]
spatial_scale = 1.0 / 8.0
pooled_featrues = roi_align.roi_align(features, filtered_rois_tensor, spatial_scale)

#====================[mask]==================================================

mask_model = mask.Mask(num_classes=80)
mask_out = mask_model(pooled_featrues)

# 마스크 loss 
mask_prediction = mask_out  # 모델이 예측한 마스크 값
mask_target = torch.rand_like(mask_out, dtype=torch.float)

# BCELoss를 사용하여 마스크 손실 계산
mask_criterion = nn.BCELoss()
mask_loss = mask_criterion(mask_prediction, mask_target)

#===================[cls, bbox]====================================================

#target_class = torch.randint(0, 81, (1,), dtype=torch.long)
#target_bbox = torch.randn(1, 4)

if datas:
    input, annotations = datas
    target_class = annotations
    target_class = torch.tensor([target_class])  # 'class' 키에 해당하는 값을 가져와서 텐서로 변환

    target_bbox = annotations


    pool_height = 16
    pool_width = 16
    num_classes = 80
    model = Classifier.Classifier(pool_height, pool_width, num_classes)
    output = model(pooled_featrues)
    mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = output

    # CrossEntropyLoss 및 SmoothL1Loss 계산
    classification_criterion = nn.CrossEntropyLoss()
    # target_class를 모델 출력과 동일한 크기로 만듭니다.
    # target_class를 모델 출력과 동일한 크기로 만듭니다.
    target_class = torch.tensor([target_class.item()])  # 스칼라 값을 가진 1D 텐서로 변환
    classification_loss = classification_criterion(mrcnn_probs, target_class)

    regression_criterion = nn.SmoothL1Loss()
    regression_loss = regression_criterion(mrcnn_bbox, target_bbox)

    # 결과 출력
    print("Classification Loss:", classification_loss.item())
    print("Regression Loss:", regression_loss.item())
    print("Mask Loss:", mask_loss.item())