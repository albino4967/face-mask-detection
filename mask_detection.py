import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import torch
from tqdm import tqdm
import pickle
import argparse

class MaskDataset(object):
    def __init__(self, transforms, path):
        '''
        path: path to train folder or test folder
        '''
        # transform module과 img path 경로를 정의
        self.transforms = transforms
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))

    def __getitem__(self, idx):  # special method
        # load images ad masks
        file_image = self.imgs[idx]
        file_label = self.imgs[idx][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)

        if 'test' in self.path:
            label_path = os.path.join("test_annotations/", file_label)
        else:
            label_path = os.path.join("annotations/", file_label)

        img = Image.open(img_path).convert("RGB")
        # Generate Label
        target = generate_target(label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def plot_image_from_output(img, annotation, file_name):
    img = img.cpu().permute(1, 2, 0)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]
        if annotation['labels'][idx] == 1:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                     facecolor='none')
        elif annotation['labels'][idx] == 2:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g',
                                     facecolor='none')
        else:
            rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange',
                                     facecolor='none')
        ax.add_patch(rect)

    plt.savefig(file_name)

def generate_box(obj):
    xmin = float(obj.find('xmin').text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    adjust_label = 1
    if obj.find('name').text == "with_mask":
        return 1 + adjust_label

    elif obj.find('name').text == "mask_weared_incorrect":
        return 2 + adjust_label

    return 0 + adjust_label

def generate_target(file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, num_epochs, data_loader, device, optimizer):
    print('----------------------train start--------------------------')
    for epoch in range(int(num_epochs)):
        start = time.time()
        model.train()
        i = 0
        epoch_loss = 0
        for imgs, annotations in tqdm(data_loader):
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses
        print(f'epoch : {epoch + 1}, Loss : {epoch_loss}, time : {time.time() - start}')

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold :
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def main(learning_rate, momentum, num_epochs, weight_decay) :
    data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor()  # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])
    current_dir = os.getcwd()
    dataset = MaskDataset(data_transform, f'{current_dir}/images/')
    test_dataset = MaskDataset(data_transform, f'{current_dir}/test_images/')
    # save

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    model = get_model_instance_segmentation(4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    train(model, num_epochs, data_loader, device, optimizer)

    with torch.no_grad():
        # 테스트셋 배치사이즈= 2
        for imgs, annotations in test_data_loader:
            imgs = list(img.to(device) for img in imgs)
            pred = make_prediction(model, imgs, 0.5)
            print(pred)
            break

    _idx = 1
    print("Target : ", annotations[_idx]['labels'])
    plot_image_from_output(imgs[_idx], annotations[_idx], "target_image.jpg")
    print("Prediction : ", pred[_idx]['labels'])
    plot_image_from_output(imgs[_idx], pred[_idx], "prediction_image.jpg")

    torch.save(model, 'mask_detection_model')

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=float, default =10, help='input epoch number')
    parser.add_argument('--weight_decay', type=float, default =0.0005, help='input 0.0001~0.001 float')
    parser.add_argument('--learning_rate', type=float, default =0.001, help='recommended 0.01')
    parser.add_argument('--momentum', type=float, default =0.9, help='recommended 0.9')
    args = parser.parse_args()

    num_epochs, weight_decay, learning_rate, momentum = args.epoch, args.weight_decay, args.learning_rate, args.momentum
    main(learning_rate, momentum, num_epochs, weight_decay)