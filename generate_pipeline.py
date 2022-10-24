from kfp import components, dsl
from kfp.components import InputPath, OutputPath
from kfp.components import func_to_container_op
from typing import NamedTuple

def download_file_from_google_drive(
        output_dataset_zipfile: OutputPath('Dataset')
):
    import gdown
    import zipfile
    import os
    zip_file_name = "mask_detection_dataset.zip"
    id_ = '1OjTtJ6I7cUtOuEBGtpccKjlWjCxiF9Bu'
    url = f'https://drive.google.com/uc?id={id_}'
    os.mkdir(output_dataset_zipfile)
    gdown.download(url, f'{output_dataset_zipfile}/{zip_file_name}', quiet=False)
    print(f'{output_dataset_zipfile}/{zip_file_name} download complete!')

    print(os.path.isdir(output_dataset_zipfile))
    print(os.path.isfile(f'{output_dataset_zipfile}/{zip_file_name}'))
    with zipfile.ZipFile(f'{output_dataset_zipfile}/{zip_file_name}', 'r') as existing_zip:
        existing_zip.extractall(output_dataset_zipfile)

    print(os.listdir(output_dataset_zipfile))

download_file_from_google_drive_op = components.create_component_from_func(
    download_file_from_google_drive, base_image='pytorch/pytorch',
    #output_component_file = 'train_data.pickle',
    packages_to_install=['gdown']
)

def data_load(
        input_dataset: InputPath('Dataset'),
        train_data: OutputPath('Dataset'),
        test_data: OutputPath('Dataset'),
):
    import os
    from PIL import Image
    from torchvision import transforms
    import torch
    from bs4 import BeautifulSoup
    import pandas as pd
    import pickle

    image_data = f'{input_dataset}/images/'
    test_image_data = f'{input_dataset}/test_images/'
    annotation = f'{input_dataset}/annotations/'
    test_annotation = f'{input_dataset}/test_annotations/'

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

    class MaskDataset(object):
        def __init__(self, transforms, path):
            '''
            path: path to train folder or test folder
            '''

            self.transforms = transforms
            self.path = path
            self.imgs = list(sorted(os.listdir(self.path)))

        def __getitem__(self, idx):  # special method
            # load images ad masks
            file_image = self.imgs[idx]
            file_label = self.imgs[idx][:-3] + 'xml'
            img_path = os.path.join(self.path, file_image)

            if 'test' in self.path:
                label_path = os.path.join(test_annotation, file_label)
            else:
                label_path = os.path.join(annotation, file_label)

            img = Image.open(img_path).convert("RGB")
            # Generate Label
            target = generate_target(label_path)

            if self.transforms is not None:
                img = self.transforms(img)

            return img, target

        def __len__(self):
            return len(self.imgs)

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = MaskDataset(data_transform, image_data)
    test_dataset = MaskDataset(data_transform, test_image_data)
    # save

    tr_data = pd.DataFrame(list(dataset))
    ts_data = pd.DataFrame(list(test_dataset))
    with open(train_data, 'wb') as f:
        pickle.dump(tr_data, f, pickle.HIGHEST_PROTOCOL)

    with open(test_data, 'wb') as f:
        pickle.dump(ts_data, f, pickle.HIGHEST_PROTOCOL)

data_load_op = components.create_component_from_func(
    data_load, base_image='pytorch/pytorch',
    packages_to_install=['pillow','torchvision', 'bs4', 'pandas']
)

def model_generation(pretrain_model : OutputPath('TFModel')):
    import torchvision
    import torch
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    torch.save(model, pretrain_model)

model_generation_op = components.create_component_from_func(
    model_generation, base_image='pytorch/pytorch',
    packages_to_install=['torchvision']
)

def train_model(
        pretrain_model: InputPath('TFModel'),
        train_data: InputPath('Dataset'),
        trained_model: OutputPath('TFModel')
) :

    def collate_fn(batch):
        return tuple(zip(*batch))

    import pickle
    import pandas as pd
    import torch
    import time
    from tqdm import tqdm

    with open(train_data, 'rb') as f:
        tr_data = pickle.load(f)

    model = torch.load(pretrain_model)

    data_loader = torch.utils.data.DataLoader(tr_data, batch_size=1, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)

    print('----------------------train start--------------------------')
    for epoch in range(5):
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

    torch.save(model, trained_model)

train_model_op = components.create_component_from_func(
    train_model, base_image='pytorch/pytorch',
    packages_to_install=['pandas']
)

def model_prediction(
        test_data: InputPath('Dataset'),
        trained_model: InputPath('TFModel')
):
    def make_prediction(model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)):
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']):
                if score > threshold:
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds

    import torch
    import pickle

    with open(test_data, 'rb') as f:
        ts_data = pickle.load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(trained_model)
    test_data_loader = torch.utils.data.DataLoader(ts_data, batch_size=1, collate_fn=collate_fn)
    with torch.no_grad():
        for imgs, annotations in test_data_loader:
            imgs = list(img.to(device) for img in imgs)
            pred = make_prediction(model, imgs, 0.5)
            print(pred)
            break

model_prediction_op = components.create_component_from_func(
    model_prediction, base_image='pytorch/pytorch',
    packages_to_install=['pandas']
)

@dsl.pipeline(name='tak test fashion mnist pipeline')
def fashion_mnist_pipeline():
    download_file_from_google_drive_op_task = download_file_from_google_drive_op()
    data_load_task = data_load_op(
        download_file_from_google_drive_op_task.outputs['output_dataset_zipfile']
    )

    model_generation_task = model_generation_op()
    train_model_task = train_model_op(model_generation_task.outputs['pretrain_model'],
                   data_load_task.outputs['train_data'])
    test_model_task = model_prediction_op(data_load_task.outputs['test_data'],
                                          train_model_task.outputs['trained_model'])

if __name__ == '__main__':
    # Compiling the pipeline
    import kfp
    kfp.compiler.Compiler().compile(fashion_mnist_pipeline, 'test.yaml')
